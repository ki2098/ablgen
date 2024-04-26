#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

typedef double complex[2];
const static int REAL = 0;
const static int IMAG = 1;

const static double PI   = M_PI;
const static int    NX   = 200;
const static int    NY   = 100;
const static int    NZ   = 100;
const static int    GC   = 3;
const static int    CX   = NX - 1;
const static int    CY   = NY - 1;
const static int    CZ   = NZ - 1;
const static int    CXYZ = CX*CY*CZ;
const static int    CCX  = CX + 2*GC;
const static int    CCY  = CY + 2*GC;
const static int    CCZ  = CZ + 2*GC;
const static int   GGXYZ = CCX*CCY*CCZ;
#define CCSIZE [CCX][CCY][CCZ]
const static double LX   = 10.;
const static double LY   = 5.;
const static double LZ   = 5.;
const static double DX   = LX/NX;
const static double DY   = LY/NY;
const static double DZ   = LZ/NZ;
const static double DXI  = 1./DX;
const static double DYI  = 1./DY;
const static double DZI  = 1./DZ;
const static double DDXI = DXI*DXI;
const static double DDYI = DYI*DYI;
const static double DDZI = DZI*DZI;
const static double DT   = 1e-3;
const static double DTI  = 1./DT;
double              MAX_CFL;

const static double UINLET      = 1.;
const static double RE          = 1e4;
const static double REI         = 1./RE;
const static double SOR_OMEGA   = 1.2;
double              MAXDIAGI   = 1.;
int                 LS_ITER;
const static int    LS_MAXITER = 1000;
const static double LS_EPS     = 1e-3;
double              LS_ERR;
int                 ISTEP;
const static double MAXT        = 10.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K;
const static double FORCING_EPS   = 1e-2;
const static double FORCING_EFK   = 1e-3;
const static double LOW_PASS      = 2.;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);

const static int NFPY = 3, NFPZ=3;
const static double XFP = 1.;
double fpposition[3][NFPY][NFPZ];
double fpforce[3][NFPY][NFPZ];

double X[CCX]={}, Y[CCY]={}, Z[CCZ]={};
double U[3][CCX][CCY][CCZ]={}, UU[3][CCX][CCY][CCZ]={}, P[CCX][CCY][CCZ]={}, UP[3][CCX][CCY][CCZ]={}, UAVG[3][CCX][CCY][CCZ];
double RHS[CCX][CCY][CCZ]={};
double FF[3][CCX][CCY][CCZ]={};
double Q[CCX][CCY][CCZ]={};
double NUT[CCX][CCY][CCZ]={};
double POIA[7][CCX][CCY][CCZ]={};

void init_env() {
    #pragma acc enter data copyin(POIA, fpposition, fpforce, U, UU, P, UP, UAVG, RHS, FF, Q, NUT, X, Y, Z)
}

void finalize_env() {
    #pragma acc exit data delete(POIA, fpposition, fpforce, U, UU, P, UP, UAVG, RHS, FF, Q, NUT, X, Y, Z)
}

struct PBiCGStab {
    double   xp[CCX][CCY][CCZ]={};
    double    r[CCX][CCY][CCZ]={};
    double   rr[CCX][CCY][CCZ]={};
    double    p[CCX][CCY][CCZ]={};
    double    q[CCX][CCY][CCZ]={};
    double    s[CCX][CCY][CCZ]={};
    double   pp[CCX][CCY][CCZ]={};
    double   ss[CCX][CCY][CCZ]={};
    double    t[CCX][CCY][CCZ]={};

    void init() {
        #pragma acc enter data copyin(this[0:1], xp, r, rr, p, q, s, pp, ss, t)
    }

    void finalize() {
        #pragma acc exit data delete(this[0:1], xp, r, rr, p, q, s, pp, ss, t)
    }
} pcg;

double gettime() {return ISTEP*DT;}

template<typename T>
inline T sqr(T a) {return a*a;}

template<typename T>
inline T cub(T a) {return a*a*a;}

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
    if ((i + j + k)%2 == color) {
        double ae1 = DDXI*max_diag_inverse;
        double aw1 = DDXI*max_diag_inverse;
        double an1 = DDYI*max_diag_inverse;
        double as1 = DDYI*max_diag_inverse;
        double at1 = DDZI*max_diag_inverse;
        double ab1 = DDZI*max_diag_inverse;
        double acc = - (ae1 + aw1 + an1 + as1 + at1 + ab1);
        double xcc = phi[i][j][k];
        double xe1 = phi[i+1][j][k];
        double xw1 = phi[i-1][j][k];
        double xn1 = phi[i][j+1][k];
        double xs1 = phi[i][j-1][k];
        double xt1 = phi[i][j][k+1];
        double xb1 = phi[i][j][k-1];
        double  cc = (rhs[i][j][k] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1)) / acc;
        phi[i][j][k] += SOR_OMEGA*cc;
        return sqr(cc);
    } else {
        return 0;
    }
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
        #pragma acc parallel loop independent reduction(+:LS_ERR) collapse(3) present(P, RHS) firstprivate(MAXDIAGI)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            LS_ERR += sor_rb_core(P, RHS, i, j, k, 1, MAXDIAGI);
        }}}
        LS_ERR = sqrt(LS_ERR/CXYZ);
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
        double xe1 =  x[i+1][j][k];
        double xw1 =  x[i-1][j][k];
        double xn1 =  x[i][j+1][k];
        double xs1 =  x[i][j-1][k];
        double xt1 =  x[i][j][k+1];
        double xb1 =  x[i][j][k-1];
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
        double xe1 =  x[i+1][j][k];
        double xw1 =  x[i-1][j][k];
        double xn1 =  x[i][j+1][k];
        double xs1 =  x[i][j-1][k];
        double xt1 =  x[i][j][k+1];
        double xb1 =  x[i][j][k-1];
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
        sum += sqr(vec[i][j][k]);
    }}}
    return sum;
}

double jacobi_core(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double xp[CCX][CCY][CCZ], double b[CCX][CCY][CCZ], int i, int j, int k) {
    double acc = a[0][i][j][k];
    double ae1 = a[1][i][j][k];
    double aw1 = a[2][i][j][k];
    double an1 = a[3][i][j][k];
    double as1 = a[4][i][j][k];
    double at1 = a[5][i][j][k];
    double ab1 = a[6][i][j][k];
    double xcc =   xp[i][j][k];
    double xe1 = xp[i+1][j][k];
    double xw1 = xp[i-1][j][k];
    double xn1 = xp[i][j+1][k];
    double xs1 = xp[i][j-1][k];
    double xt1 = xp[i][j][k+1];
    double xb1 = xp[i][j][k-1];
    double cc = (b[i][j][k] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1)) / acc;
    x[i][j][k] = xcc + cc;
    return cc*cc;
}

void copy(double dst[CCX][CCY][CCZ], double src[CCX][CCY][CCZ]) {
    #pragma acc parallel loop independent collapse(3) present(dst, src)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
        dst[i][j][k] = src[i][j][k];
    }}}
}

void jacobi_preconditioner(double a[7][CCX][CCY][CCZ], double x[CCX][CCY][CCZ], double xp[CCX][CCY][CCZ], double b[CCX][CCY][CCZ], int maxiter) {
    for (int iter = 1; iter <= maxiter; iter ++) {
        copy(xp, x);
        #pragma acc parallel loop independent collapse(3) present(a, x, xp, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            jacobi_core(a, x, xp, b, i, j, k);
        }}}
    }
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
        double xe1 =  x[i+1][j][k];
        double xw1 =  x[i-1][j][k];
        double xn1 =  x[i][j+1][k];
        double xs1 =  x[i][j-1][k];
        double xt1 =  x[i][j][k+1];
        double xb1 =  x[i][j][k-1];
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
        #pragma acc parallel loop independent collapse(3) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            rbsor_core(a, x, b, i, j, k, 1);
        }}}
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
        // jacobi_preconditioner(POIA, pcg.pp, pcg.xp, pcg.p, 5);
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
        // jacobi_preconditioner(POIA, pcg.ss, pcg.xp, pcg.s, 5);
        calc_ax(POIA, pcg.ss, pcg.t);

        omega = dot_product(pcg.t, pcg.s)/dot_product(pcg.t, pcg.t);

        #pragma acc parallel loop independent collapse(3) present(x, pcg, pcg.pp, pcg.ss) firstprivate(alpha, omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            x[i][j][k] += alpha*pcg.pp[i][j][k] + omega*pcg.ss[i][j][k];
        }}}

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
        if (i > GC) {
            acc -= DDXI;
            aw1  = DDXI;
        }
        if (i < GC+CX-1) {
            acc -= DDXI;
            ae1  = DDXI;
        }
        if (j > GC) {
            acc -= DDYI;
            as1  = DDYI;
        }
        if (j < GC+CY-1) {
            acc -= DDYI;
            an1  = DDYI;
        }
        if (k > GC) {
            acc -= DDZI;
            ab1  = DDZI;
        }
        if (k < GC+CZ-1) {
            acc -= DDZI;
            at1  = DDZI;
        }
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
    double avg = sum / double(CXYZ);
    #pragma acc parallel loop independent collapse(3) present(P) firstprivate(avg)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        P[i][j][k] -= avg;
    }}}
}

void projection_center() {
    #pragma acc kernels loop independent collapse(3) present(U, P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        U[0][i][j][k] -= .5*DT*DXI*(P[i+1][j][k] - P[i-1][j][k]);
        U[1][i][j][k] -= .5*DT*DYI*(P[i][j+1][k] - P[i][j-1][k]);
        U[2][i][j][k] -= .5*DT*DZI*(P[i][j][k+1] - P[i][j][k-1]);
    }}}
}

void projection_interface() {
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[0][i][j][k] -= DT*DXI*(P[i+1][j][k] - P[i][j][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[1][i][j][k] -= DT*DYI*(P[i][j+1][k] - P[i][j][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC-1; k < GC+CZ; k ++) {
        UU[2][i][j][k] -= DT*DZI*(P[i][j][k+1] - P[i][j][k]);
    }}}
    RMS_DIV = 0;
    #pragma acc kernels loop independent reduction(+:RMS_DIV) collapse(3) present(UU)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double dsq = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        dsq       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        dsq       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RMS_DIV   += sqr(dsq);
    }}}
    RMS_DIV = sqrt(RMS_DIV/(CXYZ));
}

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
        Q[i][j][k]  = - .5*(sqr(dudx) + sqr(dvdy) + sqr(dwdz) + 2.*(dudy*dvdx + dudz*dwdx + dvdz*dwdy));
    }}}
}

void pressure_bc() {
    #pragma acc parallel loop independent collapse(2) present(P)
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        P[GC- 1][j][k] = P[GC     ][j][k];
        P[GC+CX][j][k] = P[GC+CX-1][j][k];
    }}
    #pragma acc parallel loop independent collapse(2) present(P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        P[i][GC- 1][k] = P[i][GC     ][k];
        P[i][GC+CY][k] = P[i][GC+CY-1][k];
    }}
    #pragma acc parallel loop independent collapse(2) present(P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        P[i][j][GC -1] = P[i][j][GC     ];
        P[i][j][GC+CZ] = P[i][j][GC+CZ-1];
    }}
}

void velocity_bc() {
    #pragma acc parallel loop independent collapse(2) present(U, UP)
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        int icc = GC-1;
        for (int offset = 0; offset < GC; offset ++) {
            U[0][icc-offset][j][k] = UINLET;
            U[1][icc-offset][j][k] = 0.;
            U[2][icc-offset][j][k] = 0.;
        }
        // icc = GC+CX;
        // int ii1 = icc-1;
        // int ii2 = icc-2;
        // double dudx = .5*DXI*(3*UP[0][icc][j][k] - 4*UP[0][ii1][j][k] + UP[0][ii2][j][k]);
        // double dvdx = .5*DXI*(3*UP[1][icc][j][k] - 4*UP[1][ii1][j][k] + UP[1][ii2][j][k]);
        // double dwdx = .5*DXI*(3*UP[2][icc][j][k] - 4*UP[2][ii1][j][k] + UP[2][ii2][j][k]);
        // double ubc_xplus = UP[0][icc][j][k] - DT*UINLET*dudx;
        // double vbc_xplus = UP[1][icc][j][k] - DT*UINLET*dvdx;
        // double wbc_xplus = UP[2][icc][j][k] - DT*UINLET*dwdx;
        // U[0][icc][j][k] = ubc_xplus;
        // U[1][icc][j][k] = vbc_xplus;
        // U[2][icc][j][k] = wbc_xplus;
        // for (int offset = 1; offset < GC; offset ++) {
        //     U[0][icc+offset][j][k] = 2*ubc_xplus - U[0][icc-offset][j][k];
        //     U[1][icc+offset][j][k] = 2*vbc_xplus - U[1][icc-offset][j][k];
        //     U[2][icc+offset][j][k] = 2*wbc_xplus - U[2][icc-offset][j][k];
        // }
        for (int offset = 0; offset < GC; offset ++) {
            icc = GC+CX+offset;
            int ii1 = icc - 1;
            int ii2 = icc - 2;
            double dudx = .5*DXI*(3*UP[0][icc][j][k] - 4*UP[0][ii1][j][k] + UP[0][ii2][j][k]);
            double dvdx = .5*DXI*(3*UP[1][icc][j][k] - 4*UP[1][ii1][j][k] + UP[1][ii2][j][k]);
            double dwdx = .5*DXI*(3*UP[2][icc][j][k] - 4*UP[2][ii1][j][k] + UP[2][ii2][j][k]);
            U[0][icc][j][k] = UP[0][icc][j][k] - DT*UINLET*dudx;
            U[1][icc][j][k] = UP[1][icc][j][k] - DT*UINLET*dvdx;
            U[2][icc][j][k] = UP[2][icc][j][k] - DT*UINLET*dwdx;
        }
    }}
    #pragma acc parallel loop independent collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        int jcc = GC-1;
        double ubc_yminus = U[0][i][jcc+1][k];
        double vbc_yminus = 0.;
        double wbc_yminus = U[2][i][jcc+1][k];
        U[0][i][jcc][k] = ubc_yminus;
        U[1][i][jcc][k] = vbc_yminus;
        U[2][i][jcc][k] = wbc_yminus;
        for (int offset = 1; offset < GC; offset ++) {
            U[0][i][jcc-offset][k] = 2*ubc_yminus - U[0][i][jcc+offset][k];
            U[1][i][jcc-offset][k] = 2*vbc_yminus - U[1][i][jcc+offset][k];
            U[2][i][jcc-offset][k] = 2*wbc_yminus - U[2][i][jcc+offset][k];
        }
        jcc = GC+CY;
        double ubc_yplus = U[0][i][jcc-1][k];
        double vbc_yplus = 0.;
        double wbc_yplus = U[2][i][jcc-1][k];
        U[0][i][jcc][k] = ubc_yplus;
        U[1][i][jcc][k] = vbc_yplus;
        U[2][i][jcc][k] = wbc_yplus;
        for (int offset = 1; offset < GC; offset ++) {
            U[0][i][jcc+offset][k] = 2*ubc_yplus - U[0][i][jcc-offset][k];
            U[1][i][jcc+offset][k] = 2*vbc_yplus - U[1][i][jcc-offset][k];
            U[2][i][jcc+offset][k] = 2*wbc_yplus - U[2][i][jcc-offset][k];
        }
        
    }}
    #pragma acc parallel loop independent collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        int kcc = GC-1;
        double ubc_zminus = U[0][i][j][kcc+1];
        double vbc_zminus = U[1][i][j][kcc+1];
        double wbc_zminus = 0.;
        U[0][i][j][kcc] = ubc_zminus;
        U[1][i][j][kcc] = vbc_zminus;
        U[2][i][j][kcc] = wbc_zminus;
        for (int offset = 1; offset < GC; offset ++) {
            U[0][i][j][kcc-offset] = 2*ubc_zminus - U[0][i][j][kcc+offset];
            U[1][i][j][kcc-offset] = 2*vbc_zminus - U[1][i][j][kcc+offset];
            U[2][i][j][kcc-offset] = 2*wbc_zminus - U[2][i][j][kcc+offset];
        }
        kcc = GC+CZ;
        double ubc_zplus = U[0][i][j][kcc-1];
        double vbc_zplus = U[1][i][j][kcc-1];
        double wbc_zplus = 0.;
        U[0][i][j][kcc] = ubc_zplus;
        U[1][i][j][kcc] = vbc_zplus;
        U[2][i][j][kcc] = wbc_zplus;
        for (int offset = 1; offset < GC; offset ++) {
            U[0][i][j][kcc+offset] = 2*ubc_zplus - U[0][i][j][kcc-offset];
            U[1][i][j][kcc+offset] = 2*vbc_zplus - U[1][i][j][kcc-offset];
            U[2][i][j][kcc+offset] = 2*wbc_zplus - U[2][i][j][kcc-offset];
        }
    }}
}

void random_lagrange_forcing() {
    random_device rd;
    default_random_engine engine(rd());
    normal_distribution<double> normal(0., 1.);
    for (int j = 0; j < NFPY; j ++) {
    for (int k = 0; k < NFPZ; k ++) {
        fpforce[0][j][k] = normal(engine)*FORCING_EPS;
        fpforce[1][j][k] = normal(engine)*FORCING_EPS;
        fpforce[2][j][k] = normal(engine)*FORCING_EPS;
    }}
    #pragma acc update device(fpforce)
    #pragma acc parallel loop independent collapse(3) present(X, Y, Z, fpposition, fpforce, FF)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double fx = 0, fy = 0, fz = 0;
        double space_eps = 2*cbrt(DX*DY*DZ);
        #pragma acc loop seq
        for (int fpj = 0; fpj < NFPY; fpj ++) {
        #pragma acc loop seq
        for (int fpk = 0; fpk < NFPZ; fpk ++) {
            double dx = X[i] - fpposition[0][fpj][fpk];
            double dy = Y[j] - fpposition[1][fpj][fpk];
            double dz = Z[k] - fpposition[2][fpj][fpk];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            double eta = 1./(cub(space_eps)*sqrt(cub(PI)))*exp(- sqr(r/space_eps));
            fx += fpforce[0][fpj][fpk]*eta;
            fy += fpforce[1][fpj][fpk]*eta;
            fz += fpforce[2][fpj][fpk]*eta;
        }}
        FF[0][i][j][k] = fx;
        FF[1][i][j][k] = fy;
        FF[2][i][j][k] = fz;
    }}}
}

void force_velocity() {
    #pragma acc parallel loop independent collapse(3) present(X, Y, Z, U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        if (X[i] > 2 && X[i] < 3) {
        if (Y[j] > 2 && Y[j] < 3) {
        if (Z[k] > 2 && Z[k] < 3) {
            U[0][i][j][k] = 0.;
            U[1][i][j][k] = 0.;
            U[2][i][j][k] = 0.;
        }}}
    }}}
}

int NKX, NKY, NKZ;

#define kidx(i,j,k) (i)*NKY*NKZ+(j)*NKZ+(k)

complex *ffk[3];

void kforce_core(complex forcek1[CX*CY*CZ], complex forcek2[CX*CY*CZ], complex forcek3[CX*CY*CZ], int i, int j, int k) {
    if (i + j + k == 0) {
        forcek1[kidx(i,j,k)][REAL] = 0.;
        forcek1[kidx(i,j,k)][IMAG] = 0.;
        forcek2[kidx(i,j,k)][REAL] = 0.;
        forcek2[kidx(i,j,k)][IMAG] = 0.;
        forcek3[kidx(i,j,k)][REAL] = 0.;
        forcek3[kidx(i,j,k)][IMAG] = 0.;
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
    double kabs = sqrt(sqr(k1) + sqr(k2) + sqr(k3));
    double Cf = sqrt(FORCING_EFK/(16*PI*sqr(sqr(kabs))*DT));
    forcek1[kidx(i,j,k)][REAL] = Cf*(k2*a3 - k3*a2);
    forcek1[kidx(i,j,k)][IMAG] = Cf*(k2*b3 - k3*b2);
    forcek2[kidx(i,j,k)][REAL] = Cf*(k3*a1 - k1*a3);
    forcek2[kidx(i,j,k)][IMAG] = Cf*(k3*b1 - k1*b3);
    forcek3[kidx(i,j,k)][REAL] = Cf*(k1*a2 - k2*a1);
    forcek3[kidx(i,j,k)][IMAG] = Cf*(k1*b2 - k2*b1);
}

void generate_force() {
    for (int i = 0; i < NKX; i ++) {
    for (int j = 0; j < NKY; j ++) {
    for (int k = 0; k < NKZ; k ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LY;
        double k3 = k*2*PI/LZ;
        double kabs = sqrt(sqr(k1) + sqr(k2) + sqr(k3));
        if (kabs <= LOW_PASS) {
            kforce_core(ffk[0], ffk[1], ffk[2], i, j, k);
        }
    }}}
    #pragma acc update device(ffk[:3][:NKX*NKY*NKZ])
    // printf("wavenumber space force generated\n");
    #pragma acc parallel loop independent collapse(3) present(FF, ffk[:3][:NKX*NKY*NKZ], NKX, NKY, NKZ)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        // FF[0][i][j][k] = 0.;
        // FF[1][i][j][k] = 0.;
        // FF[2][i][j][k] = 0.;
        double fx = 0., fy = 0., fz = 0.;
        int I1 = i - GC;
        int I2 = j - GC;
        int I3 = k - GC;
        #pragma acc loop seq
        for (int K1 = 0; K1 < NKX; K1 ++) {
        #pragma acc loop seq
        for (int K2 = 0; K2 < NKY; K2 ++) {
        #pragma acc loop seq
        for (int K3 = 0; K3 < NKZ; K3 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double th3 = - 2.*PI*I3*K3/double(CZ);
            double Real = cos(th1 + th2 + th3);
            double Imag = sin(th1 + th2 + th3);
            fx += ffk[0][kidx(K1,K2,K3)][REAL]*Real - ffk[0][kidx(K1,K2,K3)][IMAG]*Imag;
            fy += ffk[1][kidx(K1,K2,K3)][REAL]*Real - ffk[1][kidx(K1,K2,K3)][IMAG]*Imag;
            fz += ffk[2][kidx(K1,K2,K3)][REAL]*Real - ffk[2][kidx(K1,K2,K3)][IMAG]*Imag;
        }}}
        FF[0][i][j][k] = fx;
        FF[1][i][j][k] = fy;
        FF[2][i][j][k] = fz;
    }}}
    // printf("physical space force generated\n");
}

void main_loop() {
// printf("-3\n");fflush(stdout);
    // force_velocity();
    copy(UP[0], U[0]);
    copy(UP[1], U[1]);
    copy(UP[2], U[2]);

    // random_lagrange_forcing();
// printf("-2\n");fflush(stdout);
    generate_force();
// printf("-1\n");fflush(stdout);
    prediction();
// printf("0\n");fflush(stdout);
    interpolation(MAXDIAGI);
// printf("1\n");fflush(stdout);
    ls_poisson();
    // pbicgstab_poisson(pcg, P);
// printf("2\n");fflush(stdout);
    pressure_centralize();
    pressure_bc();

    projection_center();
    velocity_bc();
    projection_interface();

    calc_q();
    max_cfl();
}

void output_field(string prefix, int n) {
    #pragma acc update self(U, P, Q, FF)
    char fname[128];
    sprintf(fname, "%s.%d", prefix.c_str(), n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,q\n");
    for (int k = GC; k < GC+CZ; k ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], Z[k], U[0][i][j][k], U[1][i][j][k], U[2][i][j][k], P[i][j][k], Q[i][j][k]);
    }}}
    fclose(file);
}

void output_coefficient_matrix(string prefix) {
    printf("POIA dummping\n");fflush(stdout);
    #pragma acc update self(POIA)
    char *fname = new char[prefix.size()+128];
    sprintf(fname, "%s", prefix.c_str());
    printf("Filename built\n"); fflush(stdout);
    FILE *file = fopen(fname, "w");
    if (!file) {
        printf("failed to open %s\n", fname); fflush(stdout);
    }
    fprintf(file, "x,y,z,a1,a2,a3,a4,a5,a6,a7\n");
    printf("start POIA dummping\n");fflush(stdout);
    for (int k = GC; k < GC+CZ; k ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", X[i], Y[j], Z[k], POIA[0][i][j][k], POIA[1][i][j][k], POIA[2][i][j][k], POIA[3][i][j][k], POIA[4][i][j][k], POIA[5][i][j][k], POIA[6][i][j][k]);
    }}}
    fclose(file);
    printf("POIA dumped\n");
    delete[] fname;
}

void make_grid() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC + 1)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC + 1)*DY;
    }
    for (int k = 0; k < CCZ; k ++) {
        Z[k] = (k - GC + 1)*DZ;
    }
}

void fill_field() {
    const double u_init[] = {UINLET, 0., 0.};
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
    for (int d = 0; d < 3; d ++) {
        U[d][i][j][k] = u_init[d];
    }}}}
}

void add_perturbation() {
    const double initial_pert = 0.1;
    #pragma omp parallel
    {

    random_device gen;
    default_random_engine engine(gen());
    normal_distribution<double> nor(0., 1.);
    #pragma omp for collapse(3)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        U[0][i][j][k] += nor(engine)*initial_pert;
        U[1][i][j][k] += nor(engine)*initial_pert;
        U[2][i][j][k] += nor(engine)*initial_pert;
    }}}

    }
}

void set_fp_position() {
    for (int j = 0; j < NFPY; j ++) {
    for (int k = 0; k < NFPZ; k ++) {
        fpposition[0][j][k] = XFP;
        fpposition[1][j][k] = (j + .5)*LY/NFPY;
        fpposition[2][j][k] = (k + .5)*LZ/NFPZ;
    }}
}

int main() {
    make_grid();
    fill_field();
    set_fp_position();

    const int MAX_KX = int(LOW_PASS*LX/(2*PI));
    const int MAX_KY = int(LOW_PASS*LY/(2*PI));
    const int MAX_KZ = int(LOW_PASS*LZ/(2*PI));
    NKX = MAX_KX + 1;
    NKY = MAX_KY + 1;
    NKZ = MAX_KZ + 1;
    printf("filtered wavenumber space %dx%dx%d\n", NKX, NKY, NKZ);

    ffk[0] = (complex*)malloc(sizeof(complex)*NKX*NKY*NKZ);
    ffk[1] = (complex*)malloc(sizeof(complex)*NKX*NKY*NKZ);
    ffk[2] = (complex*)malloc(sizeof(complex)*NKX*NKY*NKZ);

    #pragma acc enter data copyin(ffk[:3][:NKX*NKY*NKZ], NKX, NKY, NKZ)
    // printf("initialized 0\n");fflush(stdout);
    init_env();
    // printf("initialized 1\n");fflush(stdout);
    pcg.init();
    // printf("initialized 2\n");fflush(stdout);
    double max_diag = make_eq();
    MAXDIAGI = 1./max_diag;
    // printf("initialized 3\n");fflush(stdout);
    printf("max diag = %lf\n", max_diag);
    output_coefficient_matrix("data/a-inflow-forcing.csv");

    // force_velocity();
    interpolation(MAXDIAGI);

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%8d, %9.5lf, %5d, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, MAX_CFL);
        fflush(stdout);
    }
    printf("\n");
    output_field("data/inflow-forcing-fiels.csv", 0);
    

    pcg.finalize();
    finalize_env();
    #pragma acc exit data delete(ffk[:3][:NKX*NKY*NKZ], NKX, NKY, NKZ)

    return 0;
}
