#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

typedef double complex[2];

using namespace std;
const static double PI = M_PI;

const static int    CX   = 64;
const static int    CY   = 64;
const static int    GC   = 3;
const static int   CXY   = CX*CY;
const static int   CCX   = CX+2*GC;
const static int   CCY   = CY+2*GC;
const static int  CCXY   = CCX*CCY;
const static double LX   = 3.2;
const static double LY   = 3.2;
const static double DX   = LX/CX;
const static double DY   = LY/CY;
const static double DXI  = 1./DX;
const static double DYI  = 1./DY;
const static double DDXI = DXI*DXI;
const static double DDYI = DYI*DYI;
const static double DT   = 1e-3;
const static double DTI  = 1./DT;

const static double RE   = 1e5;
const static double REI  = 1./RE;

const static double SOR_OMEGA   = 1.2;
int                 LS_ITER;
const static int    LS_MAXITER = 1000;
const static double LS_EPS     = 1e-3;
double              LS_ERR;
int                 ISTEP;
const static double MAXT        = 500.;
const static double STATIC_AVG_START =200.;
const static double OUPUT_START = 400.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K, TURB_K_AVG=0.;
double              TURB_I, TURB_I_AVG=0.;
int                 TAVG_NSTEP = 0;
int                 STATIC_NSTEP = 0;

const static double LOW_PASS = 5.;
const static double FORCING_EFK = 1e-1;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);

const static int REAL = 0;
const static int IMAG = 1;

double MAX_CFL;

double X[CCX]={};
double Y[CCY]={};
double U[2][CCX][CCY]={};
double UU[2][CCX][CCY]={};
double UP[2][CCX][CCY]={};
double UAVG[2][CCX][CCY]={};
double UPER[2][CCX][CCY]={};
double P[CCX][CCY]={};
double RHS[CCX][CCY]={};
double FF[2][CCX][CCY]={};
double Q[CCX][CCY]={};
double POIA[5][CCX][CCY];

void init_env() {
    #pragma acc enter data copyin(U, UU, P, UP, UAVG, UPER, RHS, FF, Q, POIA)
}

void finalize_env() {
    #pragma acc exit data delete(U, UU, P, UP, UAVG, UPER, RHS, FF, Q, POIA)
}

double gettime() {return ISTEP*DT;}

inline int CIDX(int i, int j) {
    return i*CY + j;
}

template<typename T>
inline T sq(T a) {return a*a;}

void periodic_bc(double phi[CCX][CCY], int margin) {
    #pragma acc parallel loop independent collapse(2) present(phi) firstprivate(margin)
    for (int j = GC; j < GC+CY; j ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[GC- 1-offset][j] = phi[GC+CX-1-offset][j];
        phi[GC+CX+offset][j] = phi[GC     +offset][j];
    }}
    #pragma acc parallel loop independent collapse(2) present(phi) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[i][GC- 1-offset] = phi[i][GC+CY-1-offset];
        phi[i][GC+CY+offset] = phi[i][GC     +offset];
    }}
}

struct LSVars {
    double    r[CCX][CCY]={};
    double   rr[CCX][CCY]={};
    double    p[CCX][CCY]={};
    double    q[CCX][CCY]={};
    double    s[CCX][CCY]={};
    double   pp[CCX][CCY]={};
    double   ss[CCX][CCY]={};
    double    t[CCX][CCY]={};

    void init() {
        #pragma acc enter data copyin(this[0:1], r, rr, p, q, s, pp, ss, t)
    }

    void finalize() {
        #pragma acc exit data delete(this[0:1], r, rr, p, q, s, pp, ss, t)
    }
} lsvar;

double advection_core(double phi[CCX][CCY], double u[CCX][CCY], double v[CCX][CCY], double uu[CCX][CCY], double vv[CCX][CCY], int i, int j) {
    double phicc = phi[i][j];
    double phie1 = phi[i+1][j];
    double phie2 = phi[i+2][j];
    double phiw1 = phi[i-1][j];
    double phiw2 = phi[i-2][j];
    double phin1 = phi[i][j+1];
    double phin2 = phi[i][j+2];
    double phis1 = phi[i][j-1];
    double phis2 = phi[i][j-2];
    double uE = uu[i  ][j];
    double uW = uu[i-1][j];
    double vN = vv[i][j  ];
    double vS = vv[i][j-1];
    double phi1xE = (- phie2 + 27.*(phie1 - phicc) + phiw1)*DXI;
    double phi1xW = (- phie1 + 27.*(phicc - phiw1) + phiw2)*DXI;
    double phi1yN = (- phin2 + 27.*(phin1 - phicc) + phis1)*DYI;
    double phi1yS = (- phin1 + 27.*(phicc - phis1) + phis2)*DYI;
    double phi4xcc = phie2 - 4.*phie1 + 6.*phicc - 4.*phiw1 + phiw2;
    double phi4ycc = phin2 - 4.*phin1 + 6.*phicc - 4.*phis1 + phis2;
    double aE = uE*phi1xE;
    double aW = uW*phi1xW;
    double aN = vN*phi1yN;
    double aS = vS*phi1yS;
    double ucc = u[i][j];
    double vcc = v[i][j];
    double adv = (.5*(aE + aW + aN + aS) + (fabs(ucc)*phi4xcc + fabs(vcc)*phi4ycc))/24.;
    return adv;
}

double diffusion_core(double phi[CCX][CCY], int i, int j) {
    double phixE = DXI*(phi[i+1][j] - phi[i][j]);
    double phixW = DXI*(phi[i][j] - phi[i-1][j]);
    double phiyN = DYI*(phi[i][j+1] - phi[i][j]);
    double phiyS = DYI*(phi[i][j] - phi[i][j-1]);
    double diffx = REI*DXI*(phixE - phixW);
    double diffy = REI*DYI*(phiyN - phiyS);
    return diffx + diffy;
}

void prediction() {
    for (int d = 0; d < 2; d ++) {
        #pragma acc parallel loop independent collapse(2) present(U, UP, UU, FF) firstprivate(d)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            double advc = advection_core(UP[d], UP[0], UP[1], UU[0], UU[1], i, j);
            double diff = diffusion_core(UP[d], i, j);
            U[d][i][j] = UP[d][i][j] + DT*(- advc + diff + FF[d][i][j]);
        }}
    }
}

void interpolation(double max_diag_inverse) {
    #pragma acc parallel loop independent collapse(2) present(UU, U)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] = .5*(U[0][i][j] + U[0][i+1][j]);
    }}
    #pragma acc parallel loop independent collapse(2) present(UU, U)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[1][i][j] = .5*(U[1][i][j] + U[1][i][j+1]);
    }}
    #pragma acc parallel loop independent collapse(2) present(UU, RHS) firstprivate(max_diag_inverse)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double div = DXI*(UU[0][i][j] - UU[0][i-1][j]);
        div       += DYI*(UU[1][i][j] - UU[1][i][j-1]);
        RHS[i][j] = DTI*div*max_diag_inverse;
    }}
}

void calc_res(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], double r[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(a, x, b, r)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double acc = a[0][i][j];
        double ae1 = a[1][i][j];
        double aw1 = a[2][i][j];
        double an1 = a[3][i][j];
        double as1 = a[4][i][j];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        double xcc = x[i][j];
        double xe1 = x[ie][j];
        double xw1 = x[iw][j];
        double xn1 = x[i][jn];
        double xs1 = x[i][js];
        r[i][j] = b[i][j] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1);
    }}
}

void calc_ax(double a[5][CCX][CCY], double x[CCX][CCY], double ax[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(a, x, ax)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double acc = a[0][i][j];
        double ae1 = a[1][i][j];
        double aw1 = a[2][i][j];
        double an1 = a[3][i][j];
        double as1 = a[4][i][j];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        double xcc = x[i][j];
        double xe1 = x[ie][j];
        double xw1 = x[iw][j];
        double xn1 = x[i][jn];
        double xs1 = x[i][js];
        ax[i][j] = acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1;
    }}
}

double dot_product(double a[CCX][CCY], double b[CCX][CCY]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(a, b)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += a[i][j]*b[i][j];
    }}
    return sum;
}

double calc_norm2sq(double vec[CCX][CCY]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(vec)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += sq(vec[i][j]);
    }}
    return sum;
}

double rbsor_core(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int i, int j, int color) {
    if ((i + j)%2 == color) {
        double acc = a[0][i][j];
        double ae1 = a[1][i][j];
        double aw1 = a[2][i][j];
        double an1 = a[3][i][j];
        double as1 = a[4][i][j];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        double xcc =    x[i][j];
        double xe1 =   x[ie][j];
        double xw1 =   x[iw][j];
        double xn1 =   x[i][jn];
        double xs1 =   x[i][js];
        double cc  = (b[i][j] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1)) / acc;
        x[i][j] = xcc + SOR_OMEGA*cc;
        return cc*cc;
    } else {
        return 0;
    }
}

void sor_poisson(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], LSVars &sorvar) {
    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 0);
        }}
        // periodic_bc(x, 1);
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 1);
        }}
        calc_res(a, x, b, sorvar.r);
        LS_ERR = sqrt(calc_norm2sq(sorvar.r)/CXY);
        if (LS_ERR < LS_EPS) {
            break;
        }
    }
}

void sor_preconditioner(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int maxiter) {
    for (int iter = 1; iter <= maxiter; iter ++) {
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 0);
        }}
        // periodic_bc(x, 1);
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 1);
        }}
        // periodic_bc(x, 1);
    }
}

void set_field(double vec[CCX][CCY], double value) {
    #pragma acc parallel loop independent collapse(2) present(vec) firstprivate(value)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        vec[i][j] = value;
    }}
}

void copy(double dst[CCX][CCY], double src[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(dst, src)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        dst[i][j] = src[i][j];
    }}
}

void pbicgstab_poisson(LSVars &pcgvar, double x[CCX][CCY]) {
    calc_res(POIA, x, RHS, pcgvar.r);
    LS_ERR = sqrt(calc_norm2sq(pcgvar.r)/CXY);
    copy(pcgvar.rr, pcgvar.r);

    double rho, rrho=1., alpha=1., beta, omega=1.;
    set_field(pcgvar.q, 0.);

    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        rho = dot_product(pcgvar.r, pcgvar.rr);
        if (fabs(rho) < __FLT_MIN__) {
            LS_ERR = fabs(rho);
            break;
        }

        if (LS_ITER == 1) {
            copy(pcgvar.p, pcgvar.r);
        } else {
            beta = (rho*alpha)/(rrho*omega);

            #pragma acc parallel loop independent collapse(2) present(pcgvar, pcgvar.p, pcgvar.q, pcgvar.r) firstprivate(beta, omega)
            for (int i = GC; i < GC+CX; i ++) {
            for (int j = GC; j < GC+CY; j ++) {
                pcgvar.p[i][j] = pcgvar.r[i][j] + beta*(pcgvar.p[i][j] - omega*pcgvar.q[i][j]);
            }}
        }
        
        set_field(pcgvar.pp, 0.);
        sor_preconditioner(POIA, pcgvar.pp, pcgvar.p, 3);
        // periodic_bc(lsvar.pp, 1);
        calc_ax(POIA, pcgvar.pp, pcgvar.q);

        alpha = rho/dot_product(pcgvar.rr, pcgvar.q);

        #pragma acc parallel loop independent collapse(2) present(pcgvar, pcgvar.s, pcgvar.q, pcgvar.r) firstprivate(alpha)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            pcgvar.s[i][j] = pcgvar.r[i][j] - alpha*pcgvar.q[i][j];
        }}

        set_field(pcgvar.ss, 0.);
        sor_preconditioner(POIA, pcgvar.ss, pcgvar.s, 3);
        // periodic_bc(lsvar.ss, 1);
        calc_ax(POIA, pcgvar.ss, pcgvar.t);

        omega = dot_product(pcgvar.t, pcgvar.s)/dot_product(pcgvar.t, pcgvar.t);

        #pragma acc parallel loop independent collapse(2) present(x, pcgvar, pcgvar.pp, pcgvar.ss) firstprivate(alpha, omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            x[i][j] += alpha*pcgvar.pp[i][j] + omega*pcgvar.ss[i][j];
        }}
        // periodic_bc(x, 1);

        #pragma acc parallel loop independent collapse(2) present(pcgvar, pcgvar.r, pcgvar.s, pcgvar.t) firstprivate(omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            pcgvar.r[i][j] = pcgvar.s[i][j] - omega*pcgvar.t[i][j];
        }}

        rrho = rho;

        LS_ERR = sqrt(calc_norm2sq(pcgvar.r)/CXY);

        if (LS_ERR < LS_EPS) {
            break;
        }
    }
}

double make_eq() {
    double max_diag = 0.;
    #pragma acc parallel loop independent reduction(max:max_diag) collapse(2) present(POIA)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double acc=0., ae1=0., aw1=0., an1=0., as1=0.;
        acc -= DDXI;
        aw1  = DDXI;
        acc -= DDXI;
        ae1  = DDXI;
        acc -= DDYI;
        as1  = DDYI;
        acc -= DDYI;
        an1  = DDYI;
        POIA[0][i][j] = acc;
        POIA[1][i][j] = ae1;
        POIA[2][i][j] = aw1;
        POIA[3][i][j] = an1;
        POIA[4][i][j] = as1;
        if (fabs(acc) > max_diag) {
            max_diag = fabs(acc);
        }
    }}
    double max_diag_inverse = 1./max_diag;
    #pragma acc parallel loop independent collapse(2) present(POIA) firstprivate(max_diag_inverse)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        POIA[0][i][j] *= max_diag_inverse;
        POIA[1][i][j] *= max_diag_inverse;
        POIA[2][i][j] *= max_diag_inverse;
        POIA[3][i][j] *= max_diag_inverse;
        POIA[4][i][j] *= max_diag_inverse;
    }}
    MAXDIAGI = max_diag_inverse;
    return max_diag;
}

void pressure_centralize() {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += P[i][j];
    }}
    double avg = sum / double(CXY);
    #pragma acc parallel loop independent collapse(2) present(P) firstprivate(avg)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        P[i][j] -= avg;
    }}
}

void projection_center() {
    #pragma acc parallel loop independent collapse(2) present(U, P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        U[0][i][j] -= .5*DT*DXI*(P[i+1][j] - P[i-1][j]);
        U[1][i][j] -= .5*DT*DYI*(P[i][j+1] - P[i][j-1]);
    }}
}

void projection_interface() {
    #pragma acc parallel loop independent collapse(2) present(UU, P)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] -= DT*DXI*(P[i+1][j] - P[i][j]);
    }}
    #pragma acc parallel loop independent collapse(2) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[1][i][j] -= DT*DYI*(P[i][j+1] - P[i][j]);
    }}
    RMS_DIV = 0;
    #pragma acc parallel loop independent reduction(+:RMS_DIV) collapse(2) present(UU)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double dsq = DXI*(UU[0][i][j] - UU[0][i-1][j]);
        dsq       += DYI*(UU[1][i][j] - UU[1][i][j-1]);
        RMS_DIV   += sq(dsq);
    }}
    RMS_DIV = sqrt(RMS_DIV/(CXY));
}

int NNX, NNY;
#define nnidx(i,j) (i)*NNY+(j)

complex *ffk[2];

void kforce_core(complex *forcek, int i, int j) {
    if (i + j == 0) {
        forcek[nnidx(i,j)][REAL] = 0.;
        forcek[nnidx(i,j)][IMAG] = 0.;
        return;
    }
    double xp = forcek[nnidx(i,j)][REAL];
    double yp = forcek[nnidx(i,j)][IMAG];
    double r  = sqrt(xp*xp + yp*yp);
    double th = atan2(yp, xp);
    double eff = 1. + r;
    double dx = GAUSS(GEN);
    double dy = GAUSS(GEN);
    double dr = dx*cos(th) + dy*sin(th);
    double dz = dx*sin(th) - dy*cos(th);
    if (dr > 0.) {
        dr /= eff;
    } else {
        dr *= eff;
    }
    dx = dr*cos(th) + dz*sin(th);
    dy = dr*sin(th) - dz*cos(th);
    forcek[nnidx(i,j)][REAL] = xp + dx*DT;
    forcek[nnidx(i,j)][IMAG] = yp + dy*DT;
}

void generate_force() {
    for (int i = 0; i < NNX; i ++) {
    for (int j = 0; j < NNY; j ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LY;
        double k3 = 0.;
        double kabs = sqrt(k1*k1 + k2*k2 + k3*k3);
        if (kabs <= LOW_PASS) {
            kforce_core(ffk[0], i, j);
            kforce_core(ffk[1], i, j);
        } 
    }}
    #pragma acc update device(ffk[:2][:NNX*NNY])
    #pragma acc parallel loop independent collapse(2) present(FF, ffk[:2][:NNX*NNY], NNX, NNY)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        FF[0][i][j] = 0.;
        FF[1][i][j] = 0.;
        int I1 = i - GC;
        int I2 = j - GC;
        for (int K1 = 0; K1 < NNX; K1 ++) {
        for (int K2 = 0; K2 < NNY; K2 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double Real = cos(th1 + th2);
            double Imag = sin(th1 + th2);
            FF[0][i][j] += (ffk[0][nnidx(K1,K2)][REAL]*Real - ffk[0][nnidx(K1,K2)][IMAG]*Imag)*FORCING_EFK;
            FF[1][i][j] += (ffk[1][nnidx(K1,K2)][REAL]*Real - ffk[1][nnidx(K1,K2)][IMAG]*Imag)*FORCING_EFK;
        }}
    }
    }
}

void max_cfl() {
    MAX_CFL = 0;
    #pragma acc parallel loop independent reduction(max:MAX_CFL) collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double u = fabs(U[0][i][j]);
        double v = fabs(U[1][i][j]);
        double cfl_local = DT*(u/DX + v/DY);
        if (cfl_local > MAX_CFL) {
            MAX_CFL = cfl_local;
        }
    }}
}

void calc_q() {
    #pragma acc parallel loop independent collapse(2) present(U, Q)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double dudx = .5*DXI*(U[0][i+1][j] - U[0][i-1][j]);
        double dudy = .5*DYI*(U[0][i][j+1] - U[0][i][j-1]);
        double dvdx = .5*DXI*(U[1][i+1][j] - U[1][i-1][j]);
        double dvdy = .5*DYI*(U[1][i][j+1] - U[1][i][j-1]);
        double dwdx = .5*DXI*(U[2][i+1][j] - U[2][i-1][j]);
        Q[i][j]  = - .5*(sq(dudx) + sq(dvdy) + 2.*(dudy*dvdx));
    }}
}

void calc_time_average(int nstep) {
    #pragma acc parallel loop independent collapse(2) present(U, UAVG) firstprivate(nstep)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        UAVG[0][i][j] = ((nstep - 1)*UAVG[0][i][j] + U[0][i][j])/nstep;
        UAVG[1][i][j] = ((nstep - 1)*UAVG[1][i][j] + U[1][i][j])/nstep;
    }}
}

void calc_u_perturbation() {
    #pragma acc parallel loop independent collapse(2) present(U, UAVG, UPER)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        UPER[0][i][j] = U[0][i][j] - UAVG[0][i][j];
        UPER[1][i][j] = U[1][i][j] - UAVG[1][i][j];
    }}
}

void calc_turb_k() {
    TURB_K = 0;
    #pragma acc parallel loop independent reduction(+:TURB_K) collapse(2) present(UPER) 
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double du = UPER[0][i][j];
        double dv = UPER[1][i][j];
        TURB_K += .5*sqrt(du*du + dv*dv);
    }}
    TURB_K /= CXY;
}

void calc_turb_i() {
    TURB_I = 0;
    #pragma acc parallel loop independent reduction(+:TURB_I) collapse(2) present(UPER, UAVG) 
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double u_ = UAVG[0][i][j];
        double v_ = UAVG[1][i][j];
        double du = UPER[0][i][j];
        double dv = UPER[1][i][j];
        TURB_I += sqrt((du*du + dv*dv)/2.)/sqrt(u_*u_ + v_*v_);
    }}
    TURB_I /= CXY;
}

void main_loop() {
    // memcpy(&UP[0][0][0][0], &U[0][0][0][0], sizeof(double)*3*CCX*CCY*CCZ);
    copy(UP[0], U[0]);
    copy(UP[1], U[1]);
    generate_force();
    prediction();
    periodic_bc(U[0], 2);
    periodic_bc(U[1], 2);
    interpolation(MAXDIAGI);

    // pbicgstab_poisson(lsvar, P);
    sor_poisson(POIA, P, RHS, lsvar);
    // ls_poisson();
    pressure_centralize();
    periodic_bc(P, 1);

    projection_center();
    projection_interface();
    periodic_bc(U[0], 2);
    periodic_bc(U[1], 2);

    // turbulence();
    // periodic_bc(NUT, 1);

    calc_time_average(ISTEP);
    calc_u_perturbation();
    calc_turb_k();
    calc_turb_i();
    calc_q();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, P, Q, UAVG, FF)
    char fname[128];
    sprintf(fname, "data/ffforce-field.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,q,ua,va,wa,f1,f2,f3\n");
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., P[i][j], Q[i][j], UAVG[0][i][j], UAVG[1][i][j], 0., FF[0][i][j], FF[1][i][j], 0.);
    }}
    fclose(file);
}

void make_grid() {
    for (int i = 0; i < GC+CX; i ++) {
        X[i] = (i - GC + .5)*DX;
    }
    for (int j = 0; j < GC+CY; j ++) {
        Y[j] = (j - GC + .5)*DY;
    }
}

void fill_field() {
    const double u_init[] = {0., 0.};
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int d = 0; d < 2; d ++) {
        U[d][i][j] = u_init[d];
    }}}
}

int main() {
    fill_field();
    const int MAX_NX = int(LOW_PASS*LX/(2*PI));
    const int MAX_NY = int(LOW_PASS*LY/(2*PI));
    NNX = MAX_NX + 1;
    NNY = MAX_NY + 1;
    printf("filtered wavenumber space %dx%d\n", NNX, NNY);

    ffk[0] = (complex*)malloc(sizeof(complex)*NNX*NNY);
    ffk[1] = (complex*)malloc(sizeof(complex)*NNX*NNY);

    #pragma acc enter data copyin(ffk[:2][:NNX*NNY], NNX, NNY)
    init_env();
    lsvar.init();
    interpolation(MAXDIAGI);
    
    make_grid();
    make_eq();
    printf("max diag=%lf\n", 1./MAXDIAGI);

    FILE *statistic_file = fopen("data/statistics.csv", "w");
    fprintf(statistic_file, "t,k,i,kAvg,iAvg\n");
    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%8d, %9.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_K_AVG, TURB_I, TURB_I_AVG, MAX_CFL);
        if (ISTEP%int(0.1/DT) == 0 && ISTEP >= int(OUPUT_START/DT)) {
            output_field(ISTEP/int(0.1/DT));
            printf("\n");
        }
        if (ISTEP >= int(STATIC_AVG_START/DT)) {
            double NTURBAVG = ISTEP - int(STATIC_AVG_START/DT) + 1;
            TURB_K_AVG = (1. - 1./NTURBAVG)*TURB_K_AVG + (1./NTURBAVG)*TURB_K;
            TURB_I_AVG = (1. - 1./NTURBAVG)*TURB_I_AVG + (1./NTURBAVG)*TURB_I;
            if (ISTEP%int(1./DT) == 0) {
                fprintf(statistic_file, "%10.5lf,%12.5e,%12.5e,%12.5e,%12.5e\n", gettime(), TURB_K, TURB_I, TURB_K_AVG, TURB_I_AVG);
            }
        }
        fflush(stdout);
    }
    printf("\n");
    fclose(statistic_file);

    #pragma acc exit data delete(ffk[:2][:NNX*NNY], NNX, NNY)
    finalize_env();
    lsvar.finalize();

    free(ffk[0]);
    free(ffk[1]);
    return 0;
}
