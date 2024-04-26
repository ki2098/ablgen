#include <stdlib.h>
#include <stdio.h>
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
const static int    CZ   = 64;
const static int    GC   = 3;
const static int    CXYZ = CX*CY*CZ;
const static int    CCX  = CX +2*GC;
const static int    CCY  = CY +2*GC;
const static int    CCZ  = CZ +2*GC;
const static double LX   = 3.2;
const static double LY   = 3.2;
const static double LZ   = 3.2;
const static double DX   = LX/CX;
const static double DY   = LY/CY;
const static double DZ   = LZ/CZ;
const static double DXI  = 1./DX;
const static double DYI  = 1./DY;
const static double DZI  = 1./DZ;
const static double DDXI = DXI*DXI;
const static double DDYI = DYI*DYI;
const static double DDZI = DZI*DZI;
const static double DT   = 1e-3;
const static double DTI  = 1./DT;

const static double RE   = 1e4;
const static double REI  = 1./RE;

const static double SOR_OMEGA   = 1.2;
int                 LS_ITER;
const static int    LS_MAXITER = 1000;
const static double LS_EPS     = 1e-3;
double              LS_ERR;
int                 ISTEP;
const static double MAXT        = 300;
const static double STATIC_START = 200.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K;
double              TURB_K_TAVG = 0.;
double              TURB_I;
int                 TAVG_NSTEP = 0;
int                 STATIC_NSTEP = 0;

const static double LOW_PASS = 2.;
const static double FORCING_EFK = 5e-3;

default_random_engine GEN;
normal_distribution<double> GAUSS(0., 1.);

const static int REAL = 0;
const static int IMAG = 1;

double MAX_CFL;

double X[CCX]={}, Y[CCY]={}, Z[CCZ]={};
double U[3][CCX][CCY][CCZ]={}, UU[3][CCX][CCY][CCZ]={}, P[CCX][CCY][CCZ]={}, UP[3][CCX][CCY][CCZ]={}, UAVG[3][CCX][CCY][CCZ];
double RHS[CCX][CCY][CCZ]={};
double FF[3][CCX][CCY][CCZ]={};
double Q[CCX][CCY][CCZ]={};
double NUT[CCX][CCY][CCZ]={};
double POIA[7][CCX][CCY][CCZ]={};

void init_env() {
    #pragma acc enter data copyin(U, UU, P, UP, UAVG, RHS, FF, Q, NUT, POIA)
}

void finalize_env() {
    #pragma acc exit data delete(U, UU, P, UP, UAVG, RHS, FF, Q, NUT, POIA)
}

struct PBiCGStab {
    double    r[CCX][CCY][CCZ]={};
    double   rr[CCX][CCY][CCZ]={};
    double    p[CCX][CCY][CCZ]={};
    double    q[CCX][CCY][CCZ]={};
    double    s[CCX][CCY][CCZ]={};
    double   pp[CCX][CCY][CCZ]={};
    double   ss[CCX][CCY][CCZ]={};
    double    t[CCX][CCY][CCZ]={};

    void init() {
        #pragma acc enter data copyin(this[0:1], r, rr, p, q, s, pp, ss, t)
    }

    void finalize() {
        #pragma acc exit data delete(this[0:1], r, rr, p, q, s, pp, ss, t)
    }
} pcg;

inline int CIDX(int i, int j, int k) {
    return i*CY*CZ + j*CZ + k;
}

double gettime() {return ISTEP*DT;}

double limit(double center, double distance, double seed) {
    distance = fabs(distance);
    if (seed < center - distance) {
        return center - distance;
    } else if (seed > center + distance) {
        return center + distance;
    } else {
        return seed;
    }
}

template<typename T>
inline T sq(T a) {return a*a;}

double advection_core(double phi[CCX][CCY][CCZ], double u[CCX][CCY][CCZ], double v[CCX][CCY][CCZ], double w[CCX][CCY][CCZ], double uu[CCX][CCY][CCZ], double vv[CCX][CCY][CCZ], double ww[CCX][CCY][CCZ], int i, int j, int k) {
    double phicc = phi[i][j][k];
    double phie1 = phi[i+1][j][k];
    double phie2 = phi[i+2][j][k];
    double phiw1 = phi[i-1][j][k];
    double phiw2 = phi[i-2][j][k];
    double phin1 = phi[i][j+1][k];
    double phin2 = phi[i][j+2][k];
    double phis1 = phi[i][j-1][k];
    double phis2 = phi[i][j-2][k];
    double phit1 = phi[i][j][k+1];
    double phit2 = phi[i][j][k+2];
    double phib1 = phi[i][j][k-1];
    double phib2 = phi[i][j][k-2];
    double uE = uu[i  ][j][k];
    double uW = uu[i-1][j][k];
    double vN = vv[i][j  ][k];
    double vS = vv[i][j-1][k];
    double wT = ww[i][j][k  ];
    double wB = ww[i][j][k-1];
    double ucc = u[i][j][k];
    double vcc = v[i][j][k];
    double wcc = w[i][j][k];
    double phi1xE = (- phie2 + 27.*(phie1 - phicc) + phiw1)*DXI;
    double phi1xW = (- phie1 + 27.*(phicc - phiw1) + phiw2)*DXI;
    double phi1yN = (- phin2 + 27.*(phin1 - phicc) + phis1)*DYI;
    double phi1yS = (- phin1 + 27.*(phicc - phis1) + phis2)*DYI;
    double phi1zT = (- phit2 + 27.*(phit1 - phicc) + phib1)*DZI;
    double phi1zB = (- phit1 + 27.*(phicc - phib1) + phib2)*DZI;
    double phi4xcc = phie2 - 4.*phie1 + 6.*phicc - 4.*phiw1 + phiw2;
    double phi4ycc = phin2 - 4.*phin1 + 6.*phicc - 4.*phis1 + phis2;
    double phi4zcc = phit2 - 4.*phit1 + 6.*phicc - 4.*phib1 + phib2;
    double aE = uE*phi1xE;
    double aW = uW*phi1xW;
    double aN = vN*phi1yN;
    double aS = vS*phi1yS;
    double aT = wT*phi1zT;
    double aB = wB*phi1zB;
    double adv = (.5*(aE + aW + aN + aS + aT + aB) + (fabs(ucc)*phi4xcc + fabs(vcc)*phi4ycc + fabs(wcc)*phi4zcc))/24.;
    return adv;
}

double diffusion_core(double phi[CCX][CCY][CCZ], double nut[CCX][CCY][CCZ], int i, int j, int k) {
    double phixE = DXI*(phi[i+1][j][k] - phi[i][j][k]);
    double phixW = DXI*(phi[i][j][k] - phi[i-1][j][k]);
    double phiyN = DYI*(phi[i][j+1][k] - phi[i][j][k]);
    double phiyS = DYI*(phi[i][j][k] - phi[i][j-1][k]);
    double phizT = DZI*(phi[i][j][k+1] - phi[i][j][k]);
    double phizB = DZI*(phi[i][j][k] - phi[i][j][k-1]);
    double nutE  =  .5*(nut[i+1][j][k] + nut[i][j][k]);
    double nutW  =  .5*(nut[i][j][k] + nut[i][j][k-1]);
    double nutN  =  .5*(nut[i][j+1][k] + nut[i][j][k]);
    double nutS  =  .5*(nut[i][j][k] + nut[i][j][k-1]);
    double nutT  =  .5*(nut[i][j][k+1] + nut[i][j][k]);
    double nutB  =  .5*(nut[i][j][k] + nut[i][j][k-1]);
    double diffx = DXI*((REI + nutE)*phixE - (REI + nutW)*phixW);
    double diffy = DYI*((REI + nutN)*phiyN - (REI + nutS)*phiyS);
    double diffz = DZI*((REI + nutT)*phizT - (REI + nutB)*phizB);
    return diffx + diffy + diffz;
}

void prediction() {
    for (int d = 0; d < 3; d ++) {
        #pragma acc parallel loop independent collapse(3) present(U, UP, UU, NUT, FF) firstprivate(d)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            double advc = advection_core(UP[d], UP[0], UP[1], UP[2], UU[0], UU[1], UU[2], i, j, k);
            double diff = diffusion_core(UP[d], NUT, i, j, k);
            U[d][i][j][k] = UP[d][i][j][k] + DT*(- advc + diff + FF[d][i][j][k]);
        }}}
    }
}

void interpolation(double max_diag_inverse) {
    // printf("%lf\n", max_diag_inverse);
    #pragma acc parallel loop independent collapse(3) present(UU, U)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[0][i][j][k] = .5*(U[0][i][j][k] + U[0][i+1][j][k]);
    }}}
    #pragma acc parallel loop independent collapse(3) present(UU, U)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[1][i][j][k] = .5*(U[1][i][j][k] + U[1][i][j+1][k]);
    }}}
    #pragma acc parallel loop independent collapse(3) present(UU, U)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC-1; k < GC+CZ; k ++) {
        UU[2][i][j][k] = .5*(U[2][i][j][k] + U[2][i][j][k+1]);
    }}}
    #pragma acc parallel loop independent collapse(3) present(UU, RHS) firstprivate(max_diag_inverse)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double div = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        div       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        div       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RHS[i][j][k] = DTI*div*max_diag_inverse;
    }}}
}

double sor_rb_core(double phi[CCX][CCY][CCZ], double rhs[CCX][CCY][CCZ], int i, int j, int k, int color, double max_diag_inverse) {
    if ((i+j+k)%2 == color) {
        double ae1 = DDXI*max_diag_inverse;
        double aw1 = DDXI*max_diag_inverse;
        double an1 = DDYI*max_diag_inverse;
        double as1 = DDYI*max_diag_inverse;
        double at1 = DDZI*max_diag_inverse;
        double ab1 = DDZI*max_diag_inverse;
        double acc = - (ae1 + aw1 + an1 + as1 + at1 + ab1);
        double  cc = rhs[i][j][k];
        cc -= ae1*phi[i+1][j][k] + aw1*phi[i-1][j][k];
        cc -= an1*phi[i][j+1][k] + as1*phi[i][j-1][k];
        cc -= at1*phi[i][j][k+1] + ab1*phi[i][j][k-1];
        cc -= acc*phi[i][j][k];
        cc /= acc;
        phi[i][j][k] += SOR_OMEGA*cc;
        return sq(cc);
    } else {
        return 0;
    }
}

void periodic_bc(double phi[CCX][CCY][CCZ], int margin) {
    #pragma acc parallel loop independent collapse(2) present(phi[:CCX][:CCY][:CCZ]) firstprivate(margin)
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[GC- 1-offset][j][k] = phi[GC+CX-1-offset][j][k];
        phi[GC+CX+offset][j][k] = phi[GC     +offset][j][k];
    }}}
    #pragma acc parallel loop independent collapse(2) present(phi[:CCX][:CCY][:CCZ]) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int k = GC; k < GC+CZ; k ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[i][GC- 1-offset][k] = phi[i][GC+CY-1-offset][k];
        phi[i][GC+CY+offset][k] = phi[i][GC     +offset][k];
    }}}
    #pragma acc parallel loop independent collapse(2) present(phi[:CCX][:CCY][:CCZ]) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[i][j][GC- 1-offset] = phi[i][j][GC+CZ-1-offset];
        phi[i][j][GC+CZ+offset] = phi[i][j][GC     +offset];
    }}}
}

void ls_poisson() {
    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        LS_ERR = 0.;
        #pragma acc parallel loop independent reduction(+:LS_ERR) collapse(3) present(P, RHS) firstprivate(MAXDIAGI)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            LS_ERR += sor_rb_core(P, RHS, i, j, k, 0, MAXDIAGI);
        }}}
        periodic_bc(P, 1);
        #pragma acc parallel loop independent reduction(+:LS_ERR) collapse(3) present(P, RHS) firstprivate(MAXDIAGI)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            LS_ERR += sor_rb_core(P, RHS, i, j, k, 1, MAXDIAGI);
        }}}
        periodic_bc(P, 1);
        LS_ERR = sqrt(LS_ERR / (CX*CY*CZ));
        if (LS_ERR < LS_EPS) {
            break;
        }
    }
}

void calc_res(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double b[CCX][CCY][CCZ], double r[CCX][CCY][CCZ]) {
    #pragma acc parallel loop independent collapse(3) present(a, x, b, r)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double acc = a[0][i][j][k];
        double ae1 = a[1][i][j][k];
        double aw1 = a[2][i][j][k];
        double an1 = a[3][i][j][k];
        double as1 = a[4][i][j][k];
        double at1 = a[5][i][j][k];
        double ab1 = a[6][i][j][k];
        double xcc =    x[i][j][k];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int kb = (k > GC     )? k-1 : GC+CZ-1;
        int kt = (k < GC+CZ-1)? k+1 : GC     ;
        double xe1 =   x[ie][j][k];
        double xw1 =   x[iw][j][k];
        double xn1 =   x[i][jn][k];
        double xs1 =   x[i][js][k];
        double xt1 =   x[i][j][kt];
        double xb1 =   x[i][j][kb];
        r[i][j][k] = b[i][j][k] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1);
    }}}
}

void calc_ax(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double ax[CCX][CCY][CCZ]) {
    #pragma acc parallel loop independent collapse(3) present(a, x, ax)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double acc = a[0][i][j][k];
        double ae1 = a[1][i][j][k];
        double aw1 = a[2][i][j][k];
        double an1 = a[3][i][j][k];
        double as1 = a[4][i][j][k];
        double at1 = a[5][i][j][k];
        double ab1 = a[6][i][j][k];
        double xcc =    x[i][j][k];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int kb = (k > GC     )? k-1 : GC+CZ-1;
        int kt = (k < GC+CZ-1)? k+1 : GC     ;
        double xe1 =   x[ie][j][k];
        double xw1 =   x[iw][j][k];
        double xn1 =   x[i][jn][k];
        double xs1 =   x[i][js][k];
        double xt1 =   x[i][j][kt];
        double xb1 =   x[i][j][kb];
        ax[i][j][k] = acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1;
    }}}
}

double dot_product(double a[CCX][CCY][CCZ], double b[CCX][CCY][CCZ]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(3) present(a, b)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        sum += a[i][j][k]*b[i][j][k];
    }}}
    return sum;
}

double calc_norm2sqr(double vec[CCX][CCY][CCZ]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(3) present(vec)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        sum += sq(vec[i][j][k]);
    }}}
    return sum;
}

double rbsor_core(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double b[CCX][CCY][CCZ], int i, int j, int k, int color) {
    if ((i + j + k)%2 == color) {
        double acc = a[0][i][j][k];
        double ae1 = a[1][i][j][k];
        double aw1 = a[2][i][j][k];
        double an1 = a[3][i][j][k];
        double as1 = a[4][i][j][k];
        double at1 = a[5][i][j][k];
        double ab1 = a[6][i][j][k];
        double xcc =    x[i][j][k];
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int kb = (k > GC     )? k-1 : GC+CZ-1;
        int kt = (k < GC+CZ-1)? k+1 : GC     ;
        double xe1 =   x[ie][j][k];
        double xw1 =   x[iw][j][k];
        double xn1 =   x[i][jn][k];
        double xs1 =   x[i][js][k];
        double xt1 =   x[i][j][kt];
        double xb1 =   x[i][j][kb];
        double cc  = (b[i][j][k] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1)) / acc;
        x[i][j][k] = xcc + SOR_OMEGA*cc;
        return cc*cc;
    } else {
        return 0;
    }
}

void sor_preconditioner(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double b[CCX][CCY][CCZ], int maxiter) {
    for (int iter = 1; iter <= maxiter; iter ++) {
        #pragma acc parallel loop independent collapse(3) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            rbsor_core(a, x, b, i, j, k, 0);
        }}}
        // periodic_bc(x, 1);
        #pragma acc parallel loop independent collapse(3) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            rbsor_core(a, x, b, i, j, k, 1);
        }}}
        // periodic_bc(x, 1);
    }
}

void set_field(double vec[CCX][CCY][CCZ], double value) {
    #pragma acc parallel loop independent collapse(3) present(vec) firstprivate(value)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
        vec[i][j][k] = value;
    }}}
}

void copy(double dst[CCX][CCY][CCZ], double src[CCX][CCY][CCZ]) {
    #pragma acc parallel loop independent collapse(3) present(dst, src)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
        dst[i][j][k] = src[i][j][k];
    }}}
}

void pbicgstab_poisson(PBiCGStab &pcg, double x[CCX][CCY][CCZ]) {
    calc_res(POIA, x, RHS, pcg.r);
    LS_ERR = sqrt(calc_norm2sqr(pcg.r)/CXYZ);
    copy(pcg.rr, pcg.r);

    double rho, rrho=1., alpha=1., beta, omega=1.;
    set_field(pcg.q, 0.);

    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        rho = dot_product(pcg.r, pcg.rr);
        if (fabs(rho) < __FLT_MIN__) {
            LS_ERR = fabs(rho);
            break;
        }

        if (LS_ITER == 1) {
            copy(pcg.p, pcg.r);
        } else {
            beta = (rho*alpha)/(rrho*omega);

            #pragma acc parallel loop independent collapse(3) present(pcg, pcg.p, pcg.q, pcg.r) firstprivate(beta, omega)
            for (int i = GC; i < GC+CX; i ++) {
            for (int j = GC; j < GC+CY; j ++) {
            for (int k = GC; k < GC+CZ; k ++) {
                pcg.p[i][j][k] = pcg.r[i][j][k] + beta*(pcg.p[i][j][k] - omega*pcg.q[i][j][k]);
            }}}
        }
        
        set_field(pcg.pp, 0.);
        sor_preconditioner(POIA, pcg.pp, pcg.p, 3);
        // periodic_bc(pcg.pp, 1);
        calc_ax(POIA, pcg.pp, pcg.q);

        alpha = rho/dot_product(pcg.rr, pcg.q);

        #pragma acc parallel loop independent collapse(3) present(pcg, pcg.s, pcg.q, pcg.r) firstprivate(alpha)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            pcg.s[i][j][k] = pcg.r[i][j][k] - alpha*pcg.q[i][j][k];
        }}}

        set_field(pcg.ss, 0.);
        sor_preconditioner(POIA, pcg.ss, pcg.s, 3);
        // periodic_bc(pcg.ss, 1);
        calc_ax(POIA, pcg.ss, pcg.t);

        omega = dot_product(pcg.t, pcg.s)/dot_product(pcg.t, pcg.t);

        #pragma acc parallel loop independent collapse(3) present(x, pcg, pcg.pp, pcg.ss) firstprivate(alpha, omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            x[i][j][k] += alpha*pcg.pp[i][j][k] + omega*pcg.ss[i][j][k];
        }}}
        // periodic_bc(x, 1);

        #pragma acc parallel loop independent collapse(3) present(pcg, pcg.r, pcg.s, pcg.t) firstprivate(omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            pcg.r[i][j][k] = pcg.s[i][j][k] - omega*pcg.t[i][j][k];
        }}}

        rrho = rho;

        LS_ERR = sqrt(calc_norm2sqr(pcg.r)/CXYZ);

        if (LS_ERR < LS_EPS) {
            break;
        }
    }
}

double make_eq() {
    double max_diag = 0.;
    #pragma acc parallel loop independent reduction(max:max_diag) collapse(3) present(POIA)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double acc=0., ae1=0., aw1=0., an1=0., as1=0., at1=0., ab1=0.;
        // if (i > GC) {
            acc -= DDXI;
            aw1  = DDXI;
        // }
        // if (i < GC+CX-1) {
            acc -= DDXI;
            ae1  = DDXI;
        // }
        // if (j > GC) {
            acc -= DDYI;
            as1  = DDYI;
        // }
        // if (j < GC+CY-1) {
            acc -= DDYI;
            an1  = DDYI;
        // }
        // if (k > GC) {
            acc -= DDZI;
            ab1  = DDZI;
        // }
        // if (k < GC+CZ-1) {
            acc -= DDZI;
            at1  = DDZI;
        // }
        POIA[0][i][j][k] = acc;
        POIA[1][i][j][k] = ae1;
        POIA[2][i][j][k] = aw1;
        POIA[3][i][j][k] = an1;
        POIA[4][i][j][k] = as1;
        POIA[5][i][j][k] = at1;
        POIA[6][i][j][k] = ab1;
        if (fabs(acc) > max_diag) {
            max_diag = fabs(acc);
        }
    }}}
    double max_diag_inverse = 1./max_diag;
    #pragma acc parallel loop independent collapse(3) present(POIA) firstprivate(max_diag_inverse)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        POIA[0][i][j][k] *= max_diag_inverse;
        POIA[1][i][j][k] *= max_diag_inverse;
        POIA[2][i][j][k] *= max_diag_inverse;
        POIA[3][i][j][k] *= max_diag_inverse;
        POIA[4][i][j][k] *= max_diag_inverse;
        POIA[5][i][j][k] *= max_diag_inverse;
        POIA[6][i][j][k] *= max_diag_inverse;
    }}}
    MAXDIAGI = max_diag_inverse;
    return max_diag;
}

void pressure_centralize() {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(3) present(P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        sum += P[i][j][k];
    }}}
    double avg = sum / double(CX*CY*CZ);
    #pragma acc parallel loop independent collapse(3) present(P) firstprivate(avg)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        P[i][j][k] -= avg;
    }}}
}

void projection_center() {
    #pragma acc parallel loop independent collapse(3) present(U, P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        U[0][i][j][k] -= .5*DT*DXI*(P[i+1][j][k] - P[i-1][j][k]);
        U[1][i][j][k] -= .5*DT*DYI*(P[i][j+1][k] - P[i][j-1][k]);
        U[2][i][j][k] -= .5*DT*DZI*(P[i][j][k+1] - P[i][j][k-1]);
    }}}
}

void projection_interface() {
    #pragma acc parallel loop independent collapse(3) present(UU, P)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[0][i][j][k] -= DT*DXI*(P[i+1][j][k] - P[i][j][k]);
    }}}
    #pragma acc parallel loop independent collapse(3) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[1][i][j][k] -= DT*DYI*(P[i][j+1][k] - P[i][j][k]);
    }}}
    #pragma acc parallel loop independent collapse(3) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC-1; k < GC+CZ; k ++) {
        UU[2][i][j][k] -= DT*DZI*(P[i][j][k+1] - P[i][j][k]);
    }}}
    RMS_DIV = 0;
    #pragma acc parallel loop independent reduction(+:RMS_DIV) collapse(3) present(UU)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double dsq = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        dsq       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        dsq       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RMS_DIV   += sq(dsq);
    }}}
    RMS_DIV = sqrt(RMS_DIV/(CX*CY*CZ));
}

void turbulence_core(double u[CCX][CCY][CCZ], double v[CCX][CCY][CCZ], double w[CCX][CCY][CCZ], double nut[CCX][CCY][CCZ], int i, int j, int k) {
    double ue = u[i+1][j][k];
    double uw = u[i-1][j][k];
    double un = u[i][j+1][k];
    double us = u[i][j-1][k];
    double ut = u[i][j][k+1];
    double ub = u[i][j][k-1];
    double ve = v[i+1][j][k];
    double vw = v[i-1][j][k];
    double vn = v[i][j+1][k];
    double vs = v[i][j-1][k];
    double vt = v[i][j][k+1];
    double vb = v[i][j][k-1];
    double we = w[i+1][j][k];
    double ww = w[i-1][j][k];
    double wn = w[i][j+1][k];
    double ws = w[i][j-1][k];
    double wt = w[i][j][k+1];
    double wb = w[i][j][k-1];
    double dudx = .5*DXI*(ue - uw);
    double dudy = .5*DYI*(un - us);
    double dudz = .5*DZI*(ut - ub);
    double dvdx = .5*DXI*(ve - vw);
    double dvdy = .5*DYI*(vn - vs);
    double dvdz = .5*DZI*(vt - vb);
    double dwdx = .5*DXI*(we - ww);
    double dwdy = .5*DYI*(wn - ws);
    double dwdz = .5*DZI*(wt - wb);
    double Du = sqrt(2*sq(dudx) + 2*sq(dvdy) + 2*sq(dwdz) + sq(dudy + dvdx) + sq(dudz + dwdx) + sq(dvdz + dwdy));
    double De = cbrt(DX*DY*DZ);
    double Lc = C_SMAGORINSKY*De;
    nut[i][j][k] = sq(Lc)*Du;
}

void turbulence() {
    #pragma acc parallel loop independent collapse(3) present(U, NUT)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        turbulence_core(U[0], U[1], U[2], NUT, i, j, k);
    }}}
}

int NNX, NNY, NNZ;

#define nnidx(i,j,k) (i)*NNY*NNZ+(j)*NNZ+(k)

complex *ffk[3];

void kforce_core(complex forcek1[CX*CY*CZ], complex forcek2[CX*CY*CZ], complex forcek3[CX*CY*CZ], int i, int j, int k) {
    if (i + j + k == 0) {
        forcek1[nnidx(i,j,k)][REAL] = 0.;
        forcek1[nnidx(i,j,k)][IMAG] = 0.;
        forcek2[nnidx(i,j,k)][REAL] = 0.;
        forcek2[nnidx(i,j,k)][IMAG] = 0.;
        forcek3[nnidx(i,j,k)][REAL] = 0.;
        forcek3[nnidx(i,j,k)][IMAG] = 0.;
        return;
    }
    double a1, a2, a3, b1, b2, b3, k1, k2, k3;
    k1 = i*2*PI/LX;
    k2 = j*2*PI/LY;
    k3 = k*2*PI/LZ;
    a1 = GAUSS(GEN);
    a2 = GAUSS(GEN);
    a3 = GAUSS(GEN);
    b1 = GAUSS(GEN);
    b2 = GAUSS(GEN);
    b3 = GAUSS(GEN);
    double kabs = sqrt(sq(k1) + sq(k2) + sq(k3));
    double Cf = sqrt(FORCING_EFK/(16*PI*sq(sq(kabs))*DT));
    forcek1[nnidx(i,j,k)][REAL] = Cf*(k2*a3 - k3*a2);
    forcek1[nnidx(i,j,k)][IMAG] = Cf*(k2*b3 - k3*b2);
    forcek2[nnidx(i,j,k)][REAL] = Cf*(k3*a1 - k1*a3);
    forcek2[nnidx(i,j,k)][IMAG] = Cf*(k3*b1 - k1*b3);
    forcek3[nnidx(i,j,k)][REAL] = Cf*(k1*a2 - k2*a1);
    forcek3[nnidx(i,j,k)][IMAG] = Cf*(k1*b2 - k2*b1);
}

void scale_complex_seq(complex *seq, double scale, int size) {
    #pragma acc parallel loop independent present(seq[:size]) firstprivate(scale, size)
    for (int i = 0; i < size; i ++) {
        seq[i][REAL] *= scale;
        seq[i][IMAG] *= scale;
    }
}

void generate_force() {
    for (int i = 0; i < NNX; i ++) {
    for (int j = 0; j < NNY; j ++) {
    for (int k = 0; k < NNZ; k ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LY;
        double k3 = k*2*PI/LZ;
        double kabs = sqrt(sq(k1) + sq(k2) + sq(k3));
        if (kabs <= LOW_PASS) {
            kforce_core(ffk[0], ffk[1], ffk[2], i, j, k);
        }
    }}}
    #pragma acc update device(ffk[:3][:NNX*NNY*NNZ])
    // printf("wavenumber space force generated\n");
    #pragma acc parallel loop independent collapse(3) present(FF, ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        FF[0][i][j][k] = 0.;
        FF[1][i][j][k] = 0.;
        FF[2][i][j][k] = 0.;
        int I1 = i - GC;
        int I2 = j - GC;
        int I3 = k - GC;
        for (int K1 = 0; K1 < NNX; K1 ++) {
        for (int K2 = 0; K2 < NNY; K2 ++) {
        for (int K3 = 0; K3 < NNZ; K3 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double th3 = - 2.*PI*I3*K3/double(CZ);
            double Real = cos(th1 + th2 + th3);
            double Imag = sin(th1 + th2 + th3);
            FF[0][i][j][k] += ffk[0][nnidx(K1,K2,K3)][REAL]*Real - ffk[0][nnidx(K1,K2,K3)][IMAG]*Imag;
            FF[1][i][j][k] += ffk[1][nnidx(K1,K2,K3)][REAL]*Real - ffk[1][nnidx(K1,K2,K3)][IMAG]*Imag;
            FF[2][i][j][k] += ffk[2][nnidx(K1,K2,K3)][REAL]*Real - ffk[2][nnidx(K1,K2,K3)][IMAG]*Imag;
        }}}
    }}}
    // printf("physical space force generated\n");
}

// void turbulence_kinetic_energy() {
//     double ksum = 0;
//     #pragma acc parallel loop independent reduction(+:ksum) collapse(3) present(U)
//     for (int i = GC; i < GC+CX; i ++) {
//     for (int j = GC; j < GC+CY; j ++) {
//     for (int k = GC; k < GC+CZ; k ++) {
//         double u = U[0][i][j][k];
//         double v = U[1][i][j][k];
//         double w = U[2][i][j][k];
//         ksum += .5*sqrt(sq(u) + sq(v) + sq(w));
//     }}}
//     TURB_K = ksum / (CX*CY*CZ);
// }

void max_cfl() {
    MAX_CFL = 0;
    #pragma acc parallel loop independent reduction(max:MAX_CFL) collapse(3) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double u = fabs(U[0][i][j][k]);
        double v = fabs(U[1][i][j][k]);
        double w = fabs(U[2][i][j][k]);
        double cfl_local = DT*(u/DX + v/DY + w/DZ);
        if (cfl_local > MAX_CFL) {
            MAX_CFL = cfl_local;
        }
    }}}
}

void calc_q() {
    #pragma acc parallel loop independent collapse(3) present(U, Q)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double dudx = .5*DXI*(U[0][i+1][j][k] - U[0][i-1][j][k]);
        double dudy = .5*DYI*(U[0][i][j+1][k] - U[0][i][j-1][k]);
        double dudz = .5*DZI*(U[0][i][j][k+1] - U[0][i][j][k-1]);
        double dvdx = .5*DXI*(U[1][i+1][j][k] - U[1][i-1][j][k]);
        double dvdy = .5*DYI*(U[1][i][j+1][k] - U[1][i][j-1][k]);
        double dvdz = .5*DZI*(U[1][i][j][k+1] - U[1][i][j][k-1]);
        double dwdx = .5*DXI*(U[2][i+1][j][k] - U[2][i-1][j][k]);
        double dwdy = .5*DYI*(U[2][i][j+1][k] - U[2][i][j-1][k]);
        double dwdz = .5*DZI*(U[2][i][j][k+1] - U[2][i][j][k-1]);
        Q[i][j][k]  = - .5*(sq(dudx) + sq(dvdy) + sq(dwdz) + 2.*(dudy*dvdx + dudz*dwdx + dvdz*dwdy));
    }}}
}

void time_accumulate() {
    #pragma acc parallel loop independent collapse(3) present(U, UAVG)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
        UAVG[0][i][j][k] += U[0][i][j][k];
        UAVG[1][i][j][k] += U[1][i][j][k];
        UAVG[2][i][j][k] += U[2][i][j][k];
    }}}
}

void turbulence_kinetic_energy() {
    TURB_K = 0;
    #pragma acc parallel loop independent reduction(+:TURB_K) collapse(3) present(U, UAVG) firstprivate(TAVG_NSTEP)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double uavg = UAVG[0][i][j][k] / TAVG_NSTEP;
        double vavg = UAVG[1][i][j][k] / TAVG_NSTEP;
        double wavg = UAVG[2][i][j][k] / TAVG_NSTEP;
        double uper = U[0][i][j][k] - uavg;
        double vper = U[1][i][j][k] - vavg;
        double wper = U[2][i][j][k] - wavg;
        TURB_K += .5*(sq(uper) + sq(vper) + sq(wper));
    }}}
    TURB_K /= (CX*CY*CZ);
    if (ISTEP >= int(STATIC_START/DT)) {
        TURB_K_TAVG += TURB_K;
        STATIC_NSTEP ++;
    }
}

void main_loop() {
    // memcpy(&UP[0][0][0][0], &U[0][0][0][0], sizeof(double)*3*CCX*CCY*CCZ);
    copy(UP[0], U[0]);
    copy(UP[1], U[1]);
    copy(UP[2], U[2]);
    generate_force();
    prediction();
    periodic_bc(U[0], 2);
    periodic_bc(U[1], 2);
    periodic_bc(U[2], 2);
    interpolation(MAXDIAGI);

    pbicgstab_poisson(pcg, P);
    // ls_poisson();
    pressure_centralize();
    periodic_bc(P, 1);

    projection_center();
    projection_interface();
    periodic_bc(U[0], 2);
    periodic_bc(U[1], 2);
    periodic_bc(U[2], 2);

    // turbulence();
    // periodic_bc(NUT, 1);

    TAVG_NSTEP ++;
    time_accumulate();
    turbulence_kinetic_energy();
    
    calc_q();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, P, Q)
    char fname[128];
    sprintf(fname, "data/ffforce-field.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,q\n");
    for (int k = GC; k < GC+CZ; k ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", X[i], Y[j], Z[k], U[0][i][j][k], U[1][i][j][k], U[2][i][j][k], P[i][j][k], Q[i][j][k]);
    }}}
    fclose(file);
}

void output_kspace_force(complex forcex[CX*CY*CZ], complex forcey[CX*CY*CZ], complex forcez[CX*CY*CZ]) {
    FILE *file = fopen("data/force-k.csv", "w");
    fprintf(file, "i,j,k,fx_real,fx_image,fx_mag,fy_real,fy_image,fy_mag,fz_real,fz_image,fz_mag\n");
    double fxr, fxi, fxm, fyr, fyi, fym, fzr, fzi, fzm;
    for (int k = 0; k < NNX; k ++) {
    for (int j = 0; j < NNY; j ++) {
    for (int i = 0; i < NNZ; i ++) {
        fxr = forcex[nnidx(i,j,k)][REAL];
        fxi = forcex[nnidx(i,j,k)][IMAG];
        fyr = forcey[nnidx(i,j,k)][REAL];
        fyi = forcey[nnidx(i,j,k)][IMAG];
        fzr = forcez[nnidx(i,j,k)][REAL];
        fzi = forcez[nnidx(i,j,k)][IMAG];
        fxm = sqrt(sq(fxr) + sq(fxi));
        fym = sqrt(sq(fyr) + sq(fyi));
        fzm = sqrt(sq(fzr) + sq(fzi));
        fprintf(file, "%5d,%5d,%5d,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", i, j, k, fxr, fxi, fxm, fyr, fyi, fym, fzr, fzi, fzm);
    }}}
    fclose(file);
}

void output_complex_force(complex forcex[CX*CY*CZ], complex forcey[CX*CY*CZ], complex forcez[CX*CY*CZ]) {
    FILE *file = fopen("data/force-complex.csv", "w");
    fprintf(file, "x,y,z,fx_real,fx_image,fx_mag,fy_real,fy_image,fy_mag,fz_real,fz_image,fz_mag\n");
    double fxr, fxi, fxm, fyr, fyi, fym, fzr, fzi, fzm;
    for (int k = 0; k < CZ; k ++) {
    for (int j = 0; j < CY; j ++) {
    for (int i = 0; i < CX; i ++) {
        fxr = forcex[CIDX(i,j,k)][REAL];
        fxi = forcex[CIDX(i,j,k)][IMAG];
        fyr = forcey[CIDX(i,j,k)][REAL];
        fyi = forcey[CIDX(i,j,k)][IMAG];
        fzr = forcez[CIDX(i,j,k)][REAL];
        fzi = forcez[CIDX(i,j,k)][IMAG];
        fxm = sqrt(sq(fxr) + sq(fxi));
        fym = sqrt(sq(fyr) + sq(fyi));
        fzm = sqrt(sq(fzr) + sq(fzi));
        fprintf(file, "%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", X[i+GC], Y[j+GC], Z[k+GC], fxr, fxi, fxm, fyr, fyi, fym, fzr, fzi, fzm);
    }}}
    fclose(file);
}

void output_force(double forcex[CCX][CCY][CCZ], double forcey[CCX][CCY][CCZ], double forcez[CCX][CCY][CCZ]) {
    FILE *file = fopen("data/force.csv", "w");
    fprintf(file, "x,y,z,fx,fy,fz\n");
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        fprintf(file, "%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", X[i], Y[j], Z[k], forcex[i][j][k], forcey[i][j][k], forcez[i][j][k]);
    }}}
    fclose(file);
}

void make_grid() {
    for (int i = 0; i < GC+CX; i ++) {
        X[i] = (i - GC + .5)*DX;
    }
    for (int j = 0; j < GC+CY; j ++) {
        Y[j] = (j - GC + .5)*DY;
    }
    for (int k = 0; k < GC+CZ; k ++) {
        Z[k] = (k - GC + .5)*DZ;
    }
}

int main() {
    const int MAX_NX = int(LOW_PASS*LX/(2*PI));
    const int MAX_NY = int(LOW_PASS*LY/(2*PI));
    const int MAX_NZ = int(LOW_PASS*LZ/(2*PI));
    NNX = MAX_NX + 1;
    NNY = MAX_NY + 1;
    NNZ = MAX_NZ + 1;
    printf("filtered wavenumber space %dx%dx%d\n", NNX, NNY, NNZ);

    ffk[0] = (complex*)malloc(sizeof(complex)*NNX*NNY*NNZ);
    ffk[1] = (complex*)malloc(sizeof(complex)*NNX*NNY*NNZ);
    ffk[2] = (complex*)malloc(sizeof(complex)*NNX*NNY*NNZ);

    #pragma acc enter data copyin(ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    init_env();
    pcg.init();
    
    make_grid();
    make_eq();
    printf("max diag=%lf\n", 1./MAXDIAGI);

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%8d, %9.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_K_TAVG/STATIC_NSTEP, MAX_CFL);
        fflush(stdout);
        if (ISTEP%int(1./DT) == 0) {
            if (ISTEP >= int(STATIC_START/DT)) {
                // output_field(ISTEP/int(1./DT));
            }
            printf("\n");
        }
    }
    printf("\n");
    output_field(0);
    TURB_K_TAVG /= STATIC_NSTEP;
    printf("time space average turbulence k=%e\n", TURB_K_TAVG);

    #pragma acc exit data delete(ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    finalize_env();
    pcg.finalize();

    free(ffk[0]);
    free(ffk[1]);
    free(ffk[2]);
    return 0;
}
