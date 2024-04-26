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
const static int    NX   = 64;
const static int    NY   = 64;
const static int    NZ   = 64;
const static int    GC   = 3;
const static int    CX   = NX;
const static int    CY   = NY;
const static int    CZ   = NZ;
const static int    CXYZ = CX*CY*CZ;
const static int    CCX  = CX + 2*GC;
const static int    CCY  = CY + 2*GC;
const static int    CCZ  = CZ + 2*GC;
const static int   CCXYZ = CCX*CCY*CCZ;
const static double LX   = 3.2;
const static double LY   = 3.2;
const static double LZ   = 3.2;
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
const static double LS_EPS     = 1e-6;
double              LS_ERR;
int                 ISTEP;
const static double MAXT        = 400.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K;
double              TURB_I;
const static double FORCING_EPS   = 1e-2;
const static double FORCING_EFK   = 1e-3;
const static double LOW_PASS      = 2.;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);

double X[CCX]={};
double Y[CCY]={};
double Z[CCZ]={};
double U[3][CCX][CCY][CCZ]={};
double UU[3][CCX][CCY][CCZ]={};
double P[CCX][CCY][CCZ]={};
double UP[3][CCX][CCY][CCZ]={};
double UAVG[3][CCX][CCY][CCZ];
double RHS[CCX][CCY][CCZ]={};
double FF[3][CCX][CCY][CCZ]={};
double Q[CCX][CCY][CCZ]={};
double NUT[CCX][CCY][CCZ]={};
double POIA[7][CCX][CCY][CCZ]={};
double DVR[CCX][CCY][CCZ]={};

const static int NFPY = 5, NFPZ=5;
const static double XFP = 1.;
double fpposition[3][NFPY][NFPZ];
double fpforce[3][NFPY][NFPZ];

void init_env() {
    #pragma acc enter data copyin(DVR, POIA, fpposition, fpforce, U, UU, P, UP, UAVG, RHS, FF, Q, NUT, X, Y, Z)
}

void finalize_env() {
    #pragma acc exit data delete(DVR, POIA, fpposition, fpforce, U, UU, P, UP, UAVG, RHS, FF, Q, NUT, X, Y, Z)
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

void copy(double dst[CCX][CCY][CCZ], double src[CCX][CCY][CCZ]) {
    #pragma acc parallel loop independent collapse(3) present(dst, src)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
        dst[i][j][k] = src[i][j][k];
    }}}
}

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
    for (int i = GC; i < GC+CX  ; i ++) {
    for (int j = GC; j < GC+CY  ; j ++) {
    for (int k = GC; k < GC+CZ-1; k ++) {
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

void xy_periodic(double phi[CCX][CCY][CCZ], int margin) {
    #pragma acc parallel loop independent collapse(2) present(phi) firstprivate(margin)
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[GC- 1-offset][j][k] = phi[GC+CX-1-offset][j][k];
        phi[GC+CX+offset][j][k] = phi[GC     +offset][j][k];
    }}}
    #pragma acc parallel loop independent collapse(2) present(phi) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int k = GC; k < GC+CZ; k ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[i][GC- 1-offset][k] = phi[i][GC+CY-1-offset][k];
        phi[i][GC+CY+offset][k] = phi[i][GC     +offset][k];
    }}}
}

double nonmatrix_rbsob_core(double x[CCX][CCY][CCZ], double rhs[CCX][CCY][CCZ], int i, int j, int k, int color, double max_diag_inverse) {
    if ((i+j+k)%2 == color) {
        double ae1 = DDXI*max_diag_inverse;
        double aw1 = DDXI*max_diag_inverse;
        double an1 = DDYI*max_diag_inverse;
        double as1 = DDYI*max_diag_inverse;
        double at1 = DDZI*max_diag_inverse;
        double ab1 = DDZI*max_diag_inverse;
        double acc = - (ae1 + aw1 + an1 + as1 + at1 + ab1);
        double xcc = x[i][j][k];
        int ie1 = (i < GC+CX-1)? (i+1) : (GC     );
        int iw1 = (i > GC     )? (i-1) : (GC+CX-1);
        int jn1 = (j < GC+CY-1)? (j+1) : (GC     );
        int js1 = (j > GC     )? (j-1) : (GC+CY-1);
        int kt1 =                 k+1;
        int kb1 =                 k-1;
        double xe1 = x[ie1][j][k];
        double xw1 = x[iw1][j][k];
        double xn1 = x[i][jn1][k];
        double xs1 = x[i][js1][k];
        double xt1 = x[i][j][kt1];
        double xb1 = x[i][j][kb1];
        double cc = (rhs[i][j][k] - (acc*xcc + ae1*xe1 + aw1*xw1 + an1*xn1 + as1*xs1 + at1*xt1 + ab1*xb1)) / acc;
        x[i][j][k] = xcc + SOR_OMEGA*cc;
        return cc*cc;
    } else {
        return 0;
    }
}

void pressure_bc();

void sor_poisson() {
    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        LS_ERR = 0.;
        #pragma acc parallel loop independent reduction(+:LS_ERR) collapse(3) present(P, RHS) firstprivate(MAXDIAGI)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            LS_ERR += nonmatrix_rbsob_core(P, RHS, i, j, k, 0, MAXDIAGI);
        }}}
        void pressure_bc();
        #pragma acc parallel loop independent reduction(+:LS_ERR) collapse(3) present(P, RHS) firstprivate(MAXDIAGI)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            LS_ERR += nonmatrix_rbsob_core(P, RHS, i, j, k, 1, MAXDIAGI);
        }}}
        LS_ERR = sqrt(LS_ERR / (CX*CY*CZ));
        if (LS_ERR < LS_EPS) {
            break;
        }
        void pressure_bc();
    }
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
    for (int i = GC; i < GC+CX  ; i ++) {
    for (int j = GC; j < GC+CY  ; j ++) {
    for (int k = GC; k < GC+CZ-1; k ++) {
        UU[2][i][j][k] -= DT*DZI*(P[i][j][k+1] - P[i][j][k]);
    }}}
    RMS_DIV = 0;
    #pragma acc kernels loop independent reduction(+:RMS_DIV) collapse(3) present(UU, DVR)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double dsq = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        dsq       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        dsq       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        DVR[i][j][k] = dsq;
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
    xy_periodic(P, 1);
    #pragma acc parallel loop independent collapse(2) present(P)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        P[i][j][GC -1] = P[i][j][GC     ];
        P[i][j][GC+CZ] = P[i][j][GC+CZ-1];
    }}
}

void velocity_bc() {
    xy_periodic(U[0], 2);
    xy_periodic(U[1], 2);
    xy_periodic(U[2], 2);
    #pragma acc parallel loop independent collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        int kcc = GC-1;
        double ubc_zminus = U[0][i][j][kcc+1];
        double vbc_zminus = U[1][i][j][kcc+1];
        double wbc_zminus = 0.;
        for (int offset = 0; offset < GC; offset ++) {
            U[0][i][j][kcc-offset] = 2*ubc_zminus - U[0][i][j][kcc+1+offset];
            U[1][i][j][kcc-offset] = 2*vbc_zminus - U[1][i][j][kcc+1+offset];
            U[2][i][j][kcc-offset] = 2*wbc_zminus - U[2][i][j][kcc+1+offset];
        }
        kcc = GC+CZ;
        double ubc_zplus = U[0][i][j][kcc-1];
        double vbc_zplus = U[1][i][j][kcc-1];
        double wbc_zplus = 0.;
        for (int offset = 0; offset < GC; offset ++) {
            U[0][i][j][kcc+offset] = 2*ubc_zplus - U[0][i][j][kcc-1-offset];
            U[1][i][j][kcc+offset] = 2*vbc_zplus - U[1][i][j][kcc-1-offset];
            U[2][i][j][kcc+offset] = 2*wbc_zplus - U[2][i][j][kcc-1-offset];
        }
    }}
}

int NNX, NNY, NNZ;

#define nnidx(i,j,k) (i)*NNY*NNZ+(j)*NNZ+(k)

complex *ffk[3];

void kforce_core(complex forcek1[CX*CY*CZ], complex forcek2[CX*CY*CZ], complex forcek3[CX*CY*CZ], int i, int j, int k) {
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
    forcek1[nnidx(i,j,k)][REAL] = Cf*(k2*a3 - k3*a2);
    forcek1[nnidx(i,j,k)][IMAG] = Cf*(k2*b3 - k3*b2);
    forcek2[nnidx(i,j,k)][REAL] = Cf*(k3*a1 - k1*a3);
    forcek2[nnidx(i,j,k)][IMAG] = Cf*(k3*b1 - k1*b3);
    forcek3[nnidx(i,j,k)][REAL] = 0./* Cf*(k1*a2 - k2*a1) */;
    forcek3[nnidx(i,j,k)][IMAG] = 0./* Cf*(k1*b2 - k2*b1) */;
}

void generate_force() {
    for (int i = 0; i < NNX; i ++) {
    for (int j = 0; j < NNY; j ++) {
    for (int k = 0; k < NNZ; k ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LY;
        double k3 = k*2*PI/LZ;
        double kabs = sqrt(sqr(k1) + sqr(k2) + sqr(k3));
        if (kabs <= LOW_PASS && (i+j+k)!=0) {
            kforce_core(ffk[0], ffk[1], ffk[2], i, j, k);
        } else {
            ffk[0][nnidx(i,j,k)][REAL] = 0.;
            ffk[0][nnidx(i,j,k)][IMAG] = 0.;
            ffk[1][nnidx(i,j,k)][REAL] = 0.;
            ffk[1][nnidx(i,j,k)][IMAG] = 0.;
            ffk[2][nnidx(i,j,k)][REAL] = 0.;
            ffk[2][nnidx(i,j,k)][IMAG] = 0.;
        }
    }}}
    #pragma acc update device(ffk[:3][:NNX*NNY*NNZ])
    // printf("wavenumber space force generated\n");
    #pragma acc parallel loop independent collapse(3) present(FF, ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double fx=0, fy=0, fz=0;
        int I1 = i - GC;
        int I2 = j - GC;
        int I3 = k - GC;
        #pragma acc loop seq
        for (int K1 = 0; K1 < NNX; K1 ++) {
        #pragma acc loop seq
        for (int K2 = 0; K2 < NNY; K2 ++) {
        #pragma acc loop seq
        for (int K3 = 0; K3 < NNZ; K3 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double th3 = - 2.*PI*I3*K3/double(CZ);
            double Real = cos(th1 + th2 + th3);
            double Imag = sin(th1 + th2 + th3);
            fx += ffk[0][nnidx(K1,K2,K3)][REAL]*Real - ffk[0][nnidx(K1,K2,K3)][IMAG]*Imag;
            fy += ffk[1][nnidx(K1,K2,K3)][REAL]*Real - ffk[1][nnidx(K1,K2,K3)][IMAG]*Imag;
            fz += ffk[2][nnidx(K1,K2,K3)][REAL]*Real - ffk[2][nnidx(K1,K2,K3)][IMAG]*Imag;
        }}}
        FF[0][i][j][k] = fx;
        FF[1][i][j][k] = fy;
        FF[2][i][j][k] = fz;
    }}}
    // printf("physical space force generated\n");
}

void calc_time_average(int nstep) {
    #pragma acc parallel loop independent collapse(3) present(U, UAVG) firstprivate(nstep)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        UAVG[0][i][j][k] = ((nstep - 1)*UAVG[0][i][j][k] + U[0][i][j][k])/nstep;
        UAVG[1][i][j][k] = ((nstep - 1)*UAVG[1][i][j][k] + U[1][i][j][k])/nstep;
        UAVG[2][i][j][k] = ((nstep - 1)*UAVG[2][i][j][k] + U[2][i][j][k])/nstep;
    }}}
}

void calc_turb_k() {
    TURB_K = 0;
    #pragma acc parallel loop independent reduction(+:TURB_K) collapse(3) present(U, UAVG) 
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double du = U[0][i][j][k] - UAVG[0][i][j][k];
        double dv = U[1][i][j][k] - UAVG[1][i][j][k];
        double dw = U[2][i][j][k] - UAVG[2][i][j][k];
        TURB_K += .5*sqrt(du*du + dv*dv + dw*dw);
    }}}
    TURB_K /= CXYZ;
}

void calc_turb_i() {
    TURB_I = 0;
    #pragma acc parallel loop independent reduction(+:TURB_I) collapse(3) present(U, UAVG) 
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double u_ = UAVG[0][i][j][k];
        double v_ = UAVG[1][i][j][k];
        double w_ = UAVG[2][i][j][k];
        double du = U[0][i][j][k] - u_;
        double dv = U[1][i][j][k] - v_;
        double dw = U[2][i][j][k] - w_;
        TURB_I += sqrt((du*du + dv*dv + dw*dw)/3.)/sqrt(u_*u_ + v_*v_ + w_*w_);
    }}}
    TURB_I /= CXYZ;
}

void main_loop() {
    copy(UP[0], U[0]);
    copy(UP[1], U[1]);
    copy(UP[2], U[2]);

    generate_force();

    prediction();
    xy_periodic(U[0], 1);
    xy_periodic(U[1], 1);
    xy_periodic(U[2], 1);

    interpolation(MAXDIAGI);

    sor_poisson();
    pressure_centralize();
    pressure_bc();

    projection_center();
    projection_interface();
    velocity_bc();

    calc_q();
    calc_time_average(ISTEP);
    calc_turb_i();
    calc_turb_k();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, P, Q, DVR, UAVG)
    char fname[128];
    sprintf(fname, "data/sowfa-style-field.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,q,divergence,uavg,vavg,wavg\n");
    for (int k = GC; k < GC+CZ; k ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], Z[k], U[0][i][j][k], U[1][i][j][k], U[2][i][j][k], P[i][j][k], Q[i][j][k], DVR[i][j][k], UAVG[0][i][j][k], UAVG[1][i][j][k], UAVG[2][i][j][k]);
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

void fill_field() {
    const double u_init[] = {UINLET, 0., 0.};
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
    for (int k = 0; k < CCZ; k ++) {
    for (int d = 0; d < 3; d ++) {
        U[d][i][j][k] = u_init[d];
    }}}}
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

    make_grid();
    fill_field();
    #pragma acc enter data copyin(ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    init_env();
    pcg.init();
    interpolation(MAXDIAGI);

    // make_eq();
    printf("max diag=%lf\n", 1./MAXDIAGI);

    FILE *statistic_file = fopen("data/statistics.csv", "w");
    fprintf(statistic_file, "t,k,i\n");

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%8d, %9.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_I, MAX_CFL);
        fflush(stdout);
        if (ISTEP%int(1./DT) == 0 && ISTEP >= int(300./DT)) {
            output_field(ISTEP/int(1./DT));
            printf("\n");
        }
        if (ISTEP%int(1./DT) == 0 && ISTEP >= int(200./DT)) {
            fprintf(statistic_file, "%10.5lf,%12.5e,%12.5e\n", gettime(), TURB_K, TURB_I);
        }
    }
    printf("\n");

    fclose(statistic_file);

    #pragma acc exit data delete(ffk[:3][:NNX*NNY*NNZ], NNX, NNY, NNZ)
    finalize_env();
    pcg.finalize();

    free(ffk[0]);
    free(ffk[1]);
    free(ffk[2]);
    return 0;
}
