#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace std;
const static double PI = M_PI;

const static int DivisionX = 300;
const static int DivisionY = 100;
const static int CX = DivisionX-1;
const static int CY = DivisionY;
const static int GC = 2;
const static int CXY = CX*CY;
const static int CCX = CX+2*GC;
const static int CCY = CY+2*GC;
const static int CCXY = CCX*CCY;

const static double LX = 15;
const static double LY = 5;
const static double DX = LX/DivisionX;
const static double DY = LY/DivisionY;
const static double DXI = 1./DX;
const static double DYI = 1./DY;
const static double DDXI = DXI*DXI;
const static double DDYI = DYI*DYI;
const static double DT = 1e-3;
const static double DTI = 1./DT;

const static double RE = 1e4;
const static double REI = 1./RE;

const static double SOR_OMEGA = 1.2;
int LS_ITER;
const static int LS_MAXITER = 1000;
const static double LS_EPS = 1e-3;
double LS_ERR;

const static int DriverSize = 100;

int ISTEP;
const static double MAXT        = 100.;
const static double OUTPUT_START = 90.;
const static double OUTPUT_INTERVAL=1.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;
double              MAX_CFL;

const static double C_SMAGORINSKY = 0.1;

void set_field(double dst[CCX][CCY], double value) {
    #pragma acc parallel loop independent collapse(2) present(dst) firstprivate(value)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        dst[i][j] = value;
    }}
}

void copy_field(double dst[CCX][CCY], double src[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(dst, src)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        dst[i][j] = src[i][j];
    }}
}

double X[CCX] = {};
double Y[CCY] = {};
double U[2*CCX*CCY] = {};
double UU[2*CCX*CCY] = {};
double UP[2*CCX*CCY] = {};
double P[CCX*CCY] = {};
double RHS[CCX*CCY] = {};
double PMAT[5*CCX*CCY] = {};
double MAP[9*CCX*CCY] = {};

void acc_init() {
    #pragma acc enter data create(X, Y, U, UU, UP, P, RHS, PMAT, MAP)
}

void acc_finalize() {
    #pragma acc exit data delete(X, Y, U, UU, UP, P, RHS, PMAT, MAP)
}

static inline int id(int i, int j, int jmax) {
    return i*jmax + j;
}

struct LSVAR {
    double   xp[CCX*CCY]={};
    double    r[CCX*CCY]={};
    double   r0[CCX*CCY]={};
    double    p[CCX*CCY]={};
    double    q[CCX*CCY]={};
    double    s[CCX*CCY]={};
    double   pp[CCX*CCY]={};
    double   ss[CCX*CCY]={};
    double    t[CCX*CCY]={};

    void init() {
        #pragma acc enter data copyin(this[:1], xp, r, r0, p, q, s, pp, ss, t)
    }

    void finalize() {
        #pragma acc exit data delete(this[:1], xp, r, r0, p, q, s, pp, ss, t)
    }
} lsvar;

void grid_init() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC + 1)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC)*DY;
    }
    #pragma acc update device(X, Y)
}

void map_init() {
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        int idcc = id(i  ,j,CCY);
        int ide1 = id(i+1,j,CCY);
        int ide2 = id(i+2,j,CCY);
        int idw1, idw2;
        if (i == GC) {
            idw1 = id(GC+DriverSize  ,j,CCY);
            idw2 = id(GC+DriverSize-1,j,CCY);
        } else {
            idw1 = id(i-1,j,CCY);
            idw2 = id(i-2,j,CCY);
        }
    }
    }
}
