#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);
uniform_real_distribution<double> UNI(0., 2*PI);

typedef double complex[2];
const static int REAL = 0;
const static int IMAG = 1;

using namespace std;
const static double PI = M_PI;

const static int GC = 2;
const static double UINFLOW = 0.5;
const static double VINFLOW = 0.;

template<typename T>
inline T sq(T a) {
    return a*a;
}

struct CommParam {
const static double RE = 1e4;
const static double REI = 1./RE;

const static double SOR_OMEGA = 1.2;
const static int LS_MAXITER = 1000;
const static double LS_EPS = 1e-3;
};

struct Precursor {
    const static int CX = 100;
    const static int CY = 100;
    const static int CXYNUM = CX*CY;
    const static int CCX = CX+2*GC;
    const static int CCY = CY+2*GC;
    const static int CCXYNUM = CCX*CCY;
    const static double LX = 5;
    const static double LY = 5;
    const static double DX = LX/CX;
    const static double DY = LY/CY;
    const static double DXI = 1./DX;
    const static double DYI = 1./DY;
    const static double DDXI = DXI*DXI;
    const static double DDYI = DYI*DYI;


    const static double LOW_PASS = 10;
    const static double HIGH_PASS = 7;
    const static double FORCING_EFK = 2;
    const static double DRAG = 1e-2*FORCING_EFK;

    double X[CCX] = {};
    double Y[CCY] = {};
    double U[2][CCX][CCY] = {};
    double OMEGA[CCX][CCY] = {};
    double OMEGAP[CCX][CCY] = {};
    double PSI[CCX][CCY] = {};
    double PMAT[5][CCX][CCY] = {};
    double RHS[CCX][CCY] = {};
    double F[CCX][CCY] = {};

    const static int NNX = int(LOW_PASS*LX/(2*PI)) + 1;
    const static int NNY = int(LOW_PASS*LY/(2*PI)) + 1;

    complex FF[NNX][NNY] = {};

    struct LSVAR {
        double   xp[CCX][CCY]={};
        double    r[CCX][CCY]={};
        double   r0[CCX][CCY]={};
        double    p[CCX][CCY]={};
        double    q[CCX][CCY]={};
        double    s[CCX][CCY]={};
        double   pp[CCX][CCY]={};
        double   ss[CCX][CCY]={};
        double    t[CCX][CCY]={};

        void init() {
            #pragma acc enter data copyin(this[:1], xp, r, r0, p, q, s, pp, ss, t)
        }

        void finalize() {
            #pragma acc exit data delete(this[:1], xp, r, r0, p, q, s, pp, ss, t)
        }
    } lsvar;

    void init_env() {
        #pragma acc enter data copyin(X, Y, U, OMEGA, OMEGAP, PSI, PMAT, RHS, F, FF)
        lsvar.init();
    }

    void finalize_env() {
        lsvar.finalize();
        #pragma acc exit data delete(X, Y, U, OMEGA, OMEGAP, PSI, PMAT, RHS, F, FF)
    }

    void copy_field(double dst[CCX][CCY], double src[CCX][CCY]) {
        #pragma acc parallel loop independent collapse(2) present(dst, src)
        for (int i = 0; i < CCX; i ++) {
        for (int j = 0; j < CCY; j ++) {
            dst[i][j] = src[i][j];
        }}
    }

    double advection_core(double phi[CCX][CCY], double u[CCX][CCY], double v[CCX][CCY], int i, int j) {
        double phicc = phi[i][j];
        double phie1 = phi[i+1][j];
        double phie2 = phi[i+2][j];
        double phiw1 = phi[i-1][j];
        double phiw2 = phi[i-2][j];
        double phin1 = phi[i][j+1];
        double phin2 = phi[i][j+2];
        double phis1 = phi[i][j-1];
        double phis2 = phi[i][j-2];
        double uc = u[i][j];
        double vc = v[i][j];
        double dx1 = uc*DXI*(- phie2 + 8*phie1 - 8*phiw1 + phiw2)/12.;
        double dx4 = fabs(uc)*DXI*.25*(phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2);
        double dy1 = vc*DYI*(- phin2 + 8*phin1 - 8*phis1 + phis2)/12.;
        double dy4 = fabs(vc)*DYI*.25*(phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2);
        return dx1 + dx4 + dy1 + dy4;
    }

    double diffusion_core(double phi[CCX][CCY], int i, int j) {
        double dx2 = DDXI*(phi[i+1][j] - 2*phi[i][j] + phi[i-1][j]);
        double dy2 = DDYI*(phi[i][j+1] - 2*phi[i][j] + phi[i][j-1]);
        return CommParam::REI*(dx2 + dy2);
    }

    void prediction();
};


void Precursor::prediction() {
    copy_field(OMEGAP, OMEGA);
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {

    }}
}
