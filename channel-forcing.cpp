#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

using namespace std;

typedef double complex[2];
const static int REAL = 0;
const static int IMAG = 1;

const static double PI   = M_PI;
const static int    CX   = 50;
const static int    CY   = 50;
const static int    CZ   = 50;
const static int    GP   = 3;
const static int    GX   = CX - 1;
const static int    GY   = CY - 1;
const static int    GZ   = CZ - 1;
const static int    GGX  = GX + 2*GP;
const static int    GGY  = GY + 2*GP;
const static int    GGZ  = GZ + 2*GP;
const static double LX   = 5.;
const static double LY   = 5.;
const static double LZ   = 5.;
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
double              MAX_CFL;


const static double RE          = 1e4;
const static double REI         = 1./RE;
const static double UINLET      = 1.;
const static double SOR_OMEGA   = 1.2;
int                 SOR_ITER;
const static int    SOR_MAXITER = 1000;
const static double SOR_EPS     = 1e-6;
double              SOR_ERR;
int                 ISTEP;
const static double MAXT        = 5e-2;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K;

const static double LOW_PASS = 2.;
const static double FORCING_EFK = 1e-6;

default_random_engine GEN;
normal_distribution<double> GAUSS(0., 1.);

double X[GGX]={}, Y[GGY]={}, Z[GGZ]={};
double U[3][GGX][GGY][GGZ]={}, UU[3][GGX][GGY][GGZ]={}, P[GGX][GGY][GGZ]={}, UP[3][GGX][GGY][GGZ]={}, UAVG[3][GGX][GGY][GGZ];
double RHS[GGX][GGY][GGZ]={};
double FF[3][GGX][GGY][GGZ]={};
double Q[GGX][GGY][GGZ]={};
double NUT[GGX][GGY][GGZ]={};

void init_env() {
    #pragma acc enter data copyin(U, UU, P, UP, UAVG, RHS, FF, Q, NUT)
}

void finialize_env() {
    #pragma acc exit data delete(U, UU, P, UP, UAVG, RHS, FF, Q, NUT)
}

double gettime() {return ISTEP*DT;}

template<typename T>
inline T sq(T a) {return a*a;}

double advection_core(double phi[GGX][GGY][GGZ], double u[GGX][GGY][GGZ], double v[GGX][GGY][GGZ], double w[GGX][GGY][GGZ], double uu[GGX][GGY][GGZ], double vv[GGX][GGY][GGZ], double ww[GGX][GGY][GGZ], int i, int j, int k) {
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

double diffusion_core(double phi[GGX][GGY][GGZ], double nut[GGX][GGY][GGZ], int i, int j, int k) {
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
        #pragma acc kernels loop independent collapse(3) present(U, UP, UU, NUT, FF)
        for (int i = GP; i < GP+GX; i ++) {
        for (int j = GP; j < GP+GY; j ++) {
        for (int k = GP; k < GP+GZ; k ++) {
            double advc = advection_core(UP[d], UP[0], UP[1], UP[2], UU[0], UU[1], UU[2], i, j, k);
            double diff = diffusion_core(UP[d], NUT, i, j, k);
            U[d][i][j][k] = UP[d][i][j][k] + DT*(- advc + diff + FF[d][i][j][k]);
        }}}
    }
}

void interpolation() {
    #pragma acc kernels loop independent collapse(3) present(UU, U)
    for (int i = GP-1; i < GP+GX; i ++) {
    for (int j = GP  ; j < GP+GY; j ++) {
    for (int k = GP  ; k < GP+GZ; k ++) {
        UU[0][i][j][k] = .5*(U[0][i][j][k] + U[0][i+1][j][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, U)
    for (int i = GP  ; i < GP+GX; i ++) {
    for (int j = GP-1; j < GP+GY; j ++) {
    for (int k = GP  ; k < GP+GZ; k ++) {
        UU[1][i][j][k] = .5*(U[1][i][j][k] + U[1][i][j+1][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, U)
    for (int i = GP  ; i < GP+GX; i ++) {
    for (int j = GP  ; j < GP+GY; j ++) {
    for (int k = GP-1; k < GP+GZ; k ++) {
        UU[2][i][j][k] = .5*(U[2][i][j][k] + U[2][i][j][k+1]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, RHS)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        double div = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        div       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        div       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RHS[i][j][k] = - DTI*div;
    }}}
}

double sor_rb_core(double phi[GGX][GGY][GGZ], double rhs[GGX][GGY][GGZ], int i, int j, int k, int color) {
    if ((i+j+k)%2 == color) {
        double phiage = (phi[i+1][j][k] + phi[i-1][j][k])*DDXI;
        phiage       += (phi[i][j+1][k] + phi[i][j-1][k])*DDYI;
        phiage       += (phi[i][j][k+1] + phi[i][j][k-1])*DDZI;
        phiage       += rhs[i][j][k];
        double dphi   = .5*phiage/(DDXI + DDYI + DDZI) - phi[i][j][k];
        phi[i][j][k] += SOR_OMEGA*dphi;
        return sq(dphi);
    } else {
        return 0;
    }
}

void ls_poisson() {
    for (SOR_ITER = 1; SOR_ITER <= SOR_MAXITER; SOR_ITER ++) {
        SOR_ERR = 0.;
        #pragma acc kernels loop independent reduction(+:SOR_ERR) collapse(3) present(P, RHS)
        for (int i = GP; i < GP+GX; i ++) {
        for (int j = GP; j < GP+GY; j ++) {
        for (int k = GP; k < GP+GZ; k ++) {
            SOR_ERR += sor_rb_core(P, RHS, i, j, k, 0);
        }}}
        #pragma acc kernels loop independent reduction(+:SOR_ERR) collapse(3) present(P, RHS)
        for (int i = GP; i < GP+GX; i ++) {
        for (int j = GP; j < GP+GY; j ++) {
        for (int k = GP; k < GP+GZ; k ++) {
            SOR_ERR += sor_rb_core(P, RHS, i, j, k, 1);
        }}}
        SOR_ERR = sqrt(SOR_ERR / (GX*GY*GZ));
        if (SOR_ERR < SOR_EPS) {
            break;
        }
    }
}

void pressure_centralize() {
    double sum = 0;
    #pragma acc kernels loop independent reduction(+:sum) collapse(3) present(P)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        sum += P[i][j][k];
    }}}
    double avg = sum / double(GX*GY*GZ);
    #pragma acc kernels loop independent collapse(3) present(P) copyin(avg)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        P[i][j][k] -= avg;
    }}}
}

void projection_center() {
    #pragma acc kernels loop independent collapse(3) present(U, P)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        U[0][i][j][k] -= .5*DT*DXI*(P[i+1][j][k] - P[i-1][j][k]);
        U[1][i][j][k] -= .5*DT*DYI*(P[i][j+1][k] - P[i][j-1][k]);
        U[2][i][j][k] -= .5*DT*DZI*(P[i][j][k+1] - P[i][j][k-1]);
    }}}
}

void projection_interface() {
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GP-1; i < GP+GX; i ++) {
    for (int j = GP  ; j < GP+GY; j ++) {
    for (int k = GP  ; k < GP+GZ; k ++) {
        UU[0][i][j][k] -= DT*DXI*(P[i+1][j][k] - P[i][j][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GP  ; i < GP+GX; i ++) {
    for (int j = GP-1; j < GP+GY; j ++) {
    for (int k = GP  ; k < GP+GZ; k ++) {
        UU[1][i][j][k] -= DT*DYI*(P[i][j+1][k] - P[i][j][k]);
    }}}
    #pragma acc kernels loop independent collapse(3) present(UU, P)
    for (int i = GP  ; i < GP+GX; i ++) {
    for (int j = GP  ; j < GP+GY; j ++) {
    for (int k = GP-1; k < GP+GZ; k ++) {
        UU[2][i][j][k] -= DT*DZI*(P[i][j][k+1] - P[i][j][k]);
    }}}
    RMS_DIV = 0;
    #pragma acc kernels loop independent reduction(+:RMS_DIV) collapse(3) present(UU)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        double dsq = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        dsq       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        dsq       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RMS_DIV   += sq(dsq);
    }}}
    RMS_DIV = sqrt(RMS_DIV/(GX*GY*GZ));
}

int NKX, NKY, NKZ;

#define kidx(i,j,k) (i)*NKY*NKZ+(j)*NKZ+(k)

complex *ffk[3];

void kforce_core(complex *forcex, complex *forcey, complex *forcez, int i, int j, int k) {
    if (i + j + k == 0) {
        forcex[kidx(i,j,k)][REAL] = 0.;
        forcex[kidx(i,j,k)][IMAG] = 0.;
        forcey[kidx(i,j,k)][REAL] = 0.;
        forcey[kidx(i,j,k)][IMAG] = 0.;
        forcez[kidx(i,j,k)][REAL] = 0.;
        forcez[kidx(i,j,k)][IMAG] = 0.;
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
    forcex[kidx(i,j,k)][REAL] = Cf*(k2*a3 - k3*a2);
    forcex[kidx(i,j,k)][IMAG] = Cf*(k2*b3 - k3*b2);
    forcey[kidx(i,j,k)][REAL] = Cf*(k3*a1 - k1*a3);
    forcey[kidx(i,j,k)][IMAG] = Cf*(k3*b1 - k1*b3);
    forcez[kidx(i,j,k)][REAL] = Cf*(k1*a2 - k2*a1);
    forcez[kidx(i,j,k)][IMAG] = Cf*(k1*b2 - k2*b1);
}

void generate_force() {
    for (int i = 0; i < NKX; i ++) {
    for (int j = 0; j < NKY; j ++) {
    for (int k = 0; k < NKZ; k ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LY;
        double k3 = k*2*PI/LZ;
        double kabs = sqrt(sq(k1) + sq(k2) + sq(k3));
        if (kabs <= LOW_PASS) {
            kforce_core(ffk[0], ffk[1], ffk[2], i, j, k);
        }
    }}}
    #pragma acc update device(ffk[:3][:NKX*NKY*NKZ])
    #pragma acc kernels loop independent collapse(3) present(FF, ffk[0:3][:NKX*NKY*NKZ], NKX, NKY, NKZ)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        FF[0][i][j][k] = 0.;
        FF[1][i][j][k] = 0.;
        FF[2][i][j][k] = 0.;
        int I1 = i - GP;
        int I2 = j - GP;
        int I3 = k - GP;
        for (int K1 = 0; K1 < NKX; K1 ++) {
        for (int K2 = 0; K2 < NKY; K2 ++) {
        for (int K3 = 0; K3 < NKZ; K3 ++) {
            double th1 = - 2.*PI*I1*K1/double(GX);
            double th2 = - 2.*PI*I2*K2/double(GY);
            double th3 = - 2.*PI*I3*K3/double(GZ);
            double Real = cos(th1 + th2 + th3);
            double Imag = sin(th1 + th2 + th3);
            FF[0][i][j][k] += ffk[0][kidx(K1,K2,K3)][REAL]*Real - ffk[0][kidx(K1,K2,K3)][IMAG]*Imag;
            FF[1][i][j][k] += ffk[1][kidx(K1,K2,K3)][REAL]*Real - ffk[1][kidx(K1,K2,K3)][IMAG]*Imag;
            FF[2][i][j][k] += ffk[2][kidx(K1,K2,K3)][REAL]*Real - ffk[2][kidx(K1,K2,K3)][IMAG]*Imag;
        }}}
    }}}
}

void max_cfl() {
    MAX_CFL = 0;
    #pragma acc kernels loop independent reduction(max:MAX_CFL) collapse(3) present(U)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
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
    #pragma acc kernels loop independent collapse(3) present(U, Q)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
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

void pressure_bc() {
    #pragma acc kernels loop independent collapse(2) present(P)
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        P[GP- 1][j][k] = P[GP     ][j][k];
        P[GP+GX][j][k] = P[GP+GX-1][j][k];
    }}
    #pragma acc kernels loop independent collapse(2) present(P)
    for (int i = GP; i < GP+GX; i ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        P[i][GP- 1][k] = P[i][GP     ][k];
        P[i][GP+GY][k] = P[i][GP+GY-1][k];
    }}
    #pragma acc kernels loop independent collapse(2) present(P)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
        P[i][j][GP -1] = P[i][j][GP     ];
        P[i][j][GP+GZ] = P[i][j][GP+GZ-1];
    }}
}

void velocity_bc() {
    #pragma acc kernels loop independent collapse(2) present(U, UP)
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        for (int offset = 0; offset < GP; offset ++) {
            U[0][GP-1-offset][j][k] = UINLET;
            U[1][GP-1-offset][j][k] = 0.;
            U[2][GP-1-offset][j][k] = 0.;
            // int icc = GP+GX+offset;
            // int ii1 = icc-1;
            // int ii2 = icc-2;
            // double dudx = .5*DXI*(3*UP[0][icc][j][k] - 4*UP[0][ii1][j][k] + UP[0][ii2][j][k]);
            // double dvdx = .5*DXI*(3*UP[1][icc][j][k] - 4*UP[1][ii1][j][k] + UP[1][ii2][j][k]);
            // double dwdx = .5*DXI*(3*UP[2][icc][j][k] - 4*UP[2][ii1][j][k] + UP[2][ii2][j][k]);
            // U[0][icc][j][k] = UP[0][icc][j][k] - DT*UINLET*dudx;
            // U[1][icc][j][k] = UP[1][icc][j][k] - DT*UINLET*dvdx;
            // U[2][icc][j][k] = UP[2][icc][j][k] - DT*UINLET*dwdx;
        }
        int icc = GP+GX;
        int ii1 = icc-1;
        int ii2 = icc-2;
        double dudx = .5*DXI*(3*UP[0][icc][j][k] - 4*UP[0][ii1][j][k] + UP[0][ii2][j][k]);
        double dvdx = .5*DXI*(3*UP[1][icc][j][k] - 4*UP[1][ii1][j][k] + UP[1][ii2][j][k]);
        double dwdx = .5*DXI*(3*UP[2][icc][j][k] - 4*UP[2][ii1][j][k] + UP[2][ii2][j][k]);
        double ubc_xplus[] = {UP[0][icc][j][k] - DT*UINLET*dudx, UP[1][icc][j][k] - DT*UINLET*dvdx, UP[2][icc][j][k] - DT*UINLET*dwdx};
        for (int d = 0; d < 3; d ++) {
            U[d][icc][j][k] = ubc_xplus[d];
            for (int offset = 1; offset < GP; offset ++) {
                U[d][icc+offset][j][k] = 2*ubc_xplus[d] - U[d][icc-offset][j][k];
            }
        }
    }}
    #pragma acc kernels loop independent collapse(2) present(U)
    for (int i = GP; i < GP+GX; i ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        double ubc_yminus[] = {U[0][i][GP][k], 0., U[2][i][GP][k]};
        for (int d = 0; d < 3; d ++) {
            U[d][i][GP-1][k] = ubc_yminus[d];
            for (int offset = 1; offset < GP; offset ++) {
                U[d][i][GP-1-offset][k] = 2*ubc_yminus[d] - U[d][i][GP-1+offset][k];
            }
        }
        double ubc_yplus[] = {U[0][i][GP+GY-1][k], 0., U[2][i][GP+GY-1][k]};
        for (int d = 0; d < 3; d ++) {
            U[d][i][GP+GY][k] = ubc_yplus[d];
            for (int offset = 1; offset < GP; offset ++) {
                U[d][i][GP+GY+offset][k] = 2*ubc_yplus[d] - U[d][i][GP+GY-offset][k];
            }
        }
    }}
    #pragma acc kernels loop independent collapse(2) present(U)
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
        double ubc_zminus[] = {U[0][i][j][GP], U[1][i][j][GP], 0.};
        for (int d = 0; d < 3; d ++) {
            U[d][i][j][GP-1] = ubc_zminus[d];
            for (int offset = 1; offset < GP; offset ++) {
                U[d][i][j][GP-1-offset] = 2*ubc_zminus[d] - U[d][i][j][GP-1+offset];
            }
        }
        double ubc_zplus[] = {U[0][i][j][GP+GZ-1], U[1][i][j][GP+GZ-1], 0.};
        for (int d = 0; d < 3; d ++) {
            U[d][i][j][GP+GZ] = ubc_zplus[d];
            for (int offset = 1; offset < GP; offset ++) {
                U[d][i][j][GP+GZ+offset] = 2*ubc_zplus[d] - U[d][i][j][GP+GZ-offset];
            }
        }
    }}
}

void main_loop() {
    #pragma acc host_data use_device(U, UP)
    cudaMemcpy(&UP[0][0][0][0], &U[0][0][0][0], sizeof(double)*3*GGX*GGY*GGZ, cudaMemcpyDeviceToDevice);
    generate_force();
    prediction();
    interpolation();

    ls_poisson();
    pressure_bc();
    pressure_centralize();

    projection_center();
    velocity_bc();
    projection_interface();

    calc_q();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, P, Q)
    char fname[128];
    sprintf(fname, "data/channel-force-field.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,q\n");
    for (int i = GP; i < GP+GX; i ++) {
    for (int j = GP; j < GP+GY; j ++) {
    for (int k = GP; k < GP+GZ; k ++) {
        fprintf(file, "%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e,%12.3e\n", X[i], Y[j], Z[k], U[0][i][j][k], U[1][i][j][k], U[2][i][j][k], P[i][j][k], Q[i][j][k]);
    }}}
    fclose(file);
}

void make_grid() {
    for (int i = 0; i < GP+GX; i ++) {
        X[i] = (i - GP + 1)*DX;
    }
    for (int j = 0; j < GP+GY; j ++) {
        Y[j] = (j - GP + 1)*DY;
    }
    for (int k = 0; k < GP+GZ; k ++) {
        Z[k] = (k - GP + 1)*DZ;
    }
}

void init_field() {
    #pragma acc kernels loop independent collapse(3) present(U)
    for (int i = 0; i < GGX; i ++) {
    for (int j = 0; j < GGY; j ++) {
    for (int k = 0; k < GGZ; k ++) {
        U[0][i][j][k] = UINLET;
        U[1][i][j][k] = 0.;
        U[2][i][j][k] = 0.;
    }}}
}

int main() {
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
    init_env();

    make_grid();
    init_field();

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%8d, %9.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), SOR_ITER, SOR_ERR, RMS_DIV, TURB_K, MAX_CFL);
        fflush(stdout);
    }
    output_field(ISTEP/int(1./DT));

    finialize_env();
    #pragma acc exit data delete(ffk[:3][:NKX*NKY*NKZ], NKX, NKY, NKZ)
}