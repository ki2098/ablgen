#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <random>
#include "fftw3.h"

using namespace std;

const static int    CX   = 16;
const static int    CY   = 16;
const static int    CZ   = 16;
const static int    GC   = 3;
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
int                 SOR_ITER;
const static int    SOR_MAXITER = 1000;
const static double SOR_EPS     = 1e-6;
double              SOR_ERR;
int                 ISTEP;
int                 MAXSTEP     = int(100./DT);
double              RMS_DIV;

const static double C_SMAGORINSKY = 0.1;

const static double FORCING_EFK = 0.01;

double X[CCX]={}, Y[CCY]={}, Z[CCZ]={};
double U[3][CCX][CCY][CCZ]={}, UU[3][CCX][CCY][CCZ]={}, P[CCX][CCY][CCZ]={}, UP[3][CCX][CCY][CCZ]={};
double RHS[CCX][CCY][CCZ]={};
double FF[3][CCX][CCY][CCZ]={};

fftw_complex FFC[3][CX*CY*CZ]={};
fftw_complex FFK[3][CX*CY*CZ]={};
const static int REAL = 0;
const static int IMAG = 1;

double NUT[CCX][CCY][CCZ]={};

random_device RD{};
mt19937_64 GEN{RD()};
normal_distribution<double> GAUSS_DISTRIBUTION_A1(0., 1.);
normal_distribution<double> GAUSS_DISTRIBUTION_A2(0., 1.);
normal_distribution<double> GAUSS_DISTRIBUTION_A3(0., 1.);
normal_distribution<double> GAUSS_DISTRIBUTION_B1(0., 1.);
normal_distribution<double> GAUSS_DISTRIBUTION_B2(0., 1.);
normal_distribution<double> GAUSS_DISTRIBUTION_B3(0., 1.);

inline int CIDX(int i, int j, int k) {
    return i*CY*CZ + j*CZ + k;
}

double gettime() {return ISTEP*DT;}

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
    memcpy(&UP[0][0][0][0], &U[0][0][0][0], sizeof(double)*3*CCX*CCY*CCZ);
    for (int d = 0; d < 3; d ++) {
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            double advc = advection_core(UP[d], UP[0], UP[1], UP[2], UU[1], UU[2], UU[3], i, j, k);
            double diff = diffusion_core(UP[d], NUT, i, j, k);
            U[d][i][j][k] = UP[d][i][j][k] + DT*(- advc + diff + FF[d][i][j][k]);
        }}}
    }
}

void interpolation() {
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[0][i][j][k] = .5*(U[0][i][j][k] + U[0][i+1][j][k]);
    }}}
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[1][i][j][k] = .5*(U[1][i][j][k] + U[1][i][j+1][k]);
    }}}
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC-1; k < GC+CZ; k ++) {
        UU[2][i][j][k] = .5*(U[2][i][j][k] + U[2][i][j][k+1]);
    }}}
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double div = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        div       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        div       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
        RHS[i][j][k] = - DTI*div;
    }}}
}

double sor_rb_core(double phi[CCX][CCY][CCZ], double rhs[CCX][CCY][CCZ], int i, int j, int k, int color) {
    if ((i+j+k)%2 == color) {
        double phiage = (phi[i+1][j][k] + phi[i-1][j][k])*DDXI;
        phiage       += (phi[i][j+1][k] + phi[i][j-1][k])*DDYI;
        phiage       += (phi[i][j][k+1] + phi[i][j][k-1])*DDZI;
        double dphi   = .5*phiage/(DDXI + DDYI + DDZI) - phi[i][j][k];
        phi[i][j][k] += SOR_OMEGA*dphi;
        return sq(dphi);
    } else {
        return 0;
    }
}

void periodic_bc(double (*phi)[CCX][CCY][CCZ], int dim, int pad) {
    for (int d = 0; d < dim; d ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
        for (int offset = 0; offset < pad; offset ++) {
            phi[d][GC -1-offset][j][k] = phi[d][GC+CX-1-offset][j][k];
            phi[d][GC+CX+offset][j][k] = phi[d][GC     +offset][j][k];
        }}}
        for (int i = GC; i < GC+CX; i ++) {
        for (int k = GC; k < GC+CZ; k ++) {
        for (int offset = 0; offset < pad; offset ++) {
            phi[d][i][GC -1-offset][k] = phi[d][i][GC+CY-1-offset][k];
            phi[d][i][GC+CY+offset][k] = phi[d][i][GC     +offset][k];
        }}}
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int offset = 0; offset < pad; offset ++) {
            phi[d][i][j][GC -1-offset] = phi[d][i][j][GC+CZ-1-offset];
            phi[d][i][j][GC+CZ+offset] = phi[d][i][j][GC     +offset];
        }}}
    }
    
}

void ls_poisson() {
    for (SOR_ITER = 1; SOR_ITER <= SOR_MAXITER; SOR_ITER ++) {
        SOR_ERR = 0.;
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            SOR_ERR += sor_rb_core(P, RHS, i, j, k, 0);
        }}}
        periodic_bc(&P, 1, 1);
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
        for (int k = GC; k < GC+CZ; k ++) {
            SOR_ERR += sor_rb_core(P, RHS, i, j, k, 1);
        }}}
        periodic_bc(&P, 1, 1);
        SOR_ERR = sqrt(SOR_ERR / (CX*CY*CZ));
        if (SOR_ERR < SOR_EPS) {
            break;
        }
    }
}

void pressure_centralize() {
    double sum = 0;
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        sum += P[i][j][k];
    }}}
    double avg = sum / double(CX*CY*CZ);
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        P[i][j][k] -= avg;
    }}}
}

void projection_center() {
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        U[0][i][j][k] -= .5*DT*DXI*(P[i+1][j][k] - P[i-1][j][k]);
        U[1][i][j][k] -= .5*DT*DYI*(P[i][j+1][k] - P[i][j-1][k]);
        U[2][i][j][k] -= .5*DT*DZI*(P[i][j][k+1] - P[i][j][k-1]);
    }}}
}

void projection_interface() {
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[0][i][j][k] -= DT*DXI*(P[i+1][j][k] - P[i][j][k]);
    }}}
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
    for (int k = GC  ; k < GC+CZ; k ++) {
        UU[1][i][j][k] -= DT*DYI*(P[i][j+1][k] - P[i][j][k]);
    }}}
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
    for (int k = GC-1; k < GC+CZ; k ++) {
        UU[2][i][j][k] -= DT*DZI*(P[i][j][k+1] - P[i][j][k]);
    }}}
    RMS_DIV = 0;
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        double dsq = DXI*(UU[0][i][j][k] - UU[0][i-1][j][k]);
        dsq       += DYI*(UU[1][i][j][k] - UU[1][i][j-1][k]);
        dsq       += DZI*(UU[2][i][j][k] - UU[2][i][j][k-1]);
    }}}
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
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int k = GC; k < GC+CZ; k ++) {
        turbulence_core(U[0], U[1], U[2], NUT, i, j, k);
    }}}
}

void kforce_core(fftw_complex forcek1[CX*CY*CZ], fftw_complex forcek2[CX*CY*CZ], fftw_complex forcek3[CX*CY*CZ], int i, int j, int k) {
    double a1, a2, a3, b1, b2, b3, k1, k2, k3;
    k1 = (i - GC)*2*M_PI/LX;
    k2 = (j - GC)*2*M_PI/LY;
    k3 = (k - GC)*2*M_PI/LZ;
    a1 = GAUSS_DISTRIBUTION_A1(GEN);
    a2 = GAUSS_DISTRIBUTION_A2(GEN);
    a3 = GAUSS_DISTRIBUTION_A3(GEN);
    b1 = GAUSS_DISTRIBUTION_B1(GEN);
    b2 = GAUSS_DISTRIBUTION_B2(GEN);
    b3 = GAUSS_DISTRIBUTION_B3(GEN);
    double kabs = sqrt(sq(k1) + sq(k2) + sq(k3));
    double Cf = sqrt(FORCING_EFK/(16*M_PI*sq(sq(kabs))*DT));
    forcek1[CIDX(i,j,k)][REAL] = k2*a3 - k3*a2;
    forcek1[CIDX(i,j,k)][IMAG] = k2*b3 - k3*b2;
    forcek2[CIDX(i,j,k)][REAL] = k3*a1 - k1*a3;
    forcek2[CIDX(i,j,k)][IMAG] = k3*b1 - k1*b3;
    forcek3[CIDX(i,j,k)][REAL] = k1*a2 - k2*a1;
    forcek3[CIDX(i,j,k)][IMAG] = k1*b2 - k2*b1;
}

void scale_complex_seq(fftw_complex *seq, double scale, int size) {
    for (int i = 0; i < size; i ++) {
        seq[i][REAL] *= scale;
        seq[i][IMAG] *= scale;
    }
}

void output_complex_force(fftw_complex forcex[CX*CY*CZ], fftw_complex forcey[CX*CY*CZ], fftw_complex forcez[CX*CY*CZ]) {
    FILE *file = fopen("force.csv", "w");
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

void generate_force() {
    for (int i = 1; i < CX; i ++) {
    for (int j = 1; j < CY; j ++) {
    for (int k = 1; k < CZ; k ++) {
        double k1 = (i - GC)*2*M_PI/LX;
        double k2 = (j - GC)*2*M_PI/LY;
        double k3 = (k - GC)*2*M_PI/LZ;
        double kabs = sqrt(sq(k1) + sq(k2) + sq(k3));
        if (kabs <= 2.) {
            kforce_core(FFK[0], FFK[1], FFK[2], i, j, k);
        }
    }}}
    // scale_complex_seq(FFK[0], 1./(CX*CY*CZ), CX*CY*CZ);
    // scale_complex_seq(FFK[1], 1./(CX*CY*CZ), CX*CY*CZ);
    // scale_complex_seq(FFK[2], 1./(CX*CY*CZ), CX*CY*CZ);
    fftw_plan planx = fftw_plan_dft_3d(CX, CY, CZ, FFK[0], FFC[0], FFTW_BACKWARD, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
    fftw_plan plany = fftw_plan_dft_3d(CX, CY, CZ, FFK[1], FFC[1], FFTW_BACKWARD, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
    fftw_plan planz = fftw_plan_dft_3d(CX, CY, CZ, FFK[2], FFC[2], FFTW_BACKWARD, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);
    fftw_execute(planx);
    fftw_execute(plany);
    fftw_execute(planz);
    fftw_destroy_plan(planx);
    fftw_destroy_plan(plany);
    fftw_destroy_plan(planz);
    
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
    make_grid();
    generate_force();
    output_complex_force(FFC[0], FFC[1], FFC[2]);

    return 0;
}
