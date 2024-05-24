#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

using namespace std;

const static double PI = M_PI;

typedef double complex[2];
const static int REAL = 0;
const static int IMAG = 1;

const static int    CX   = 100;
const static int    CY   = 100;
const static int    GC   = 3;
const static int   CXY   = CX*CY;
const static int   CCX   = CX+2*GC;
const static int   CCY   = CY+2*GC;
const static int  CCXY   = CCX*CCY;
const static double LX   = 5.;
const static double LY   = 5.;
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
const static double MAXT        = 1000.;
const static double STATIC_START =200.;
const static double OUPUT_START = 900.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K, TURB_K_AVG=0.;
double              TURB_I, TURB_I_AVG=0.;
int                 TAVG_NSTEP = 0;
int                 STATIC_NSTEP = 0;

const static double LOW_PASS = 10.;
const static double HIGH_PASS = 5.;
const static double FORCING_EFK = 10.;
const static double K = 2e-1;

const static double UINFLOW = 0.;
const static double VINFLOW = 0.;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);
uniform_real_distribution<double> UNI(0., 2*PI);

double MAX_CFL;

double X[CCX]={};
double Y[CCY]={};
double U[2][CCX][CCY]={};
double OMG[CCX][CCY]={};
double OMGP[CCX][CCY]={};
double PSI[CCX][CCY]={};
double POIA[5][CCX][CCY]={};
double RHS[CCX][CCY]={};
double FF[CCX][CCY]={};

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

void init_env() {
    #pragma acc enter data copyin(X, Y, U, OMG, OMGP, PSI, POIA, RHS, FF)
}

void finalize_env() {
    #pragma acc exit data delete(X, Y, U, OMG, OMGP, PSI, POIA, RHS, FF)
}

double gettime() {
    return ISTEP*DT;
}

template<typename T>
inline T sq(T a) {
    return a*a;
}

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
    double ucc   = u[i][j];
    double vcc   = v[i][j];
    double dx1   = ucc*(- phie2 + 8*phie1 - 8*phiw1 + phiw2);
    double dx4   = 3*fabs(ucc)*(phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2);
    double dy1   = vcc*(- phin2 + 8*phin1 - 8*phis1 + phis2);
    double dy4   = 3*fabs(vcc)*(phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2);
    return ((dx1 + dx4)*DXI + (dy1 + dy4)*DYI)/12;
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
    #pragma acc host_data use_device(OMGP, OMG)
    cudaMemcpy(&OMGP[0][0], &OMG[0][0], sizeof(double)*CCXY, cudaMemcpyDeviceToDevice);
    #pragma acc parallel loop independent collapse(2) present(OMG, OMGP, U, FF, RHS) firstprivate(MAXDIAGI)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double advc = advection_core(OMGP, U[0], U[1], i, j);
        double diff = diffusion_core(OMGP, i, j);
        OMG[i][j] = OMGP[i][j] + DT*(- advc + diff + FF[i][j] - K*OMGP[i][j]);
        RHS[i][j] = - OMG[i][j]*MAXDIAGI;
    }}
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

double calc_norm2sq(double vec[CCX][CCY]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(vec)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += sq(vec[i][j]);
    }}
    return sum;
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

void psi2u() {
    #pragma acc parallel loop independent collapse(2) present(PSI, U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double psie1 = PSI[i+1][j];
        double psie2 = PSI[i+2][j];
        double psiw1 = PSI[i-1][j];
        double psiw2 = PSI[i-2][j];
        double psin1 = PSI[i][j+1];
        double psin2 = PSI[i][j+2];
        double psis1 = PSI[i][j-1];
        double psis2 = PSI[i][j-2];
        double dpsidx = (- psie2 + 8*psie1 - 8*psiw1 + psiw2)/(12*DX);
        double dpsidy = (- psin2 + 8*psin1 - 8*psis1 + psis2)/(12*DY);
        U[0][i][j] =   dpsidy + UINFLOW;
        U[1][i][j] = - dpsidx + VINFLOW;
    }}
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

void psi_centralize() {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(PSI)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += PSI[i][j];
    }}
    double avg = sum / double(CXY);
    #pragma acc parallel loop independent collapse(2) present(PSI) firstprivate(avg)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        PSI[i][j] -= avg;
    }}
}

int NNX, NNY;
#define nnidx(i,j) (i)*NNY+(j)
complex *ffk;

void kforce_core(complex *f, int i, int j) {
    f[nnidx(i,j)][REAL] = cos(UNI(GEN))*FORCING_EFK;
    f[nnidx(i,j)][IMAG] = sin(UNI(GEN))*FORCING_EFK;
}

void generate_force() {
    for (int i = 0; i < NNX; i ++) {
    for (int j = 0; j < NNY; j ++) {
        double k1 = i*2*PI/LX;
        double k2 = j*2*PI/LX;
        double kabs = sqrt(k1*k1 + k2*k2);
        if (kabs >= HIGH_PASS && kabs <= LOW_PASS) {
            kforce_core(ffk, i, j);
        }
    }}
    #pragma acc update device(ffk[0:NNX*NNY])
    #pragma acc parallel loop independent collapse(2) present(FF, ffk[0:NNX*NNY]) firstprivate(NNX, NNY)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
            FF[i][j] = 0;
            int I1 = i - GC;
            int I2 = j - GC;
            for (int K1 = 0; K1 < NNX; K1 ++) {
            for (int K2 = 0; K2 < NNY; K2 ++) {
                double th1 = - 2*PI*I1*K1/double(CX);
                double th2 = - 2*PI*I2*K2/double(CY);
                double Real = cos(th1 + th2);
                double Imag = sin(th1 + th2);
                FF[i][j] += ffk[nnidx(K1, K2)][REAL]*Real - ffk[nnidx(K1, K2)][IMAG]*Imag;
            }}
    }}
}

void max_cfl() {
    MAX_CFL = 0;
    #pragma acc parallel loop independent reduction(max:MAX_CFL) collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double u = fabs(U[0][i][j]);
        double v = fabs(U[1][i][j]);
        double local_cfl = DT*(u/DX + v/DY);
        if (local_cfl > MAX_CFL) {
            MAX_CFL = local_cfl;
        }
    }}
}

void calc_turb_k() {
    TURB_K = 0;
    #pragma acc parallel loop independent reduction(+:TURB_K) collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double du = U[0][i][j] - UINFLOW;
        double dv = U[1][i][j] - VINFLOW;
        TURB_K += .5*sqrt(du*du + dv*dv);
    }}
    TURB_K /= CXY;
}

void calc_divergence() {
    RMS_DIV = 0;
    #pragma acc parallel loop independent reduction(+:RMS_DIV) collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double ue1 = U[0][i+1][j];
        double ue2 = U[0][i+2][j];
        double uw1 = U[0][i-1][j];
        double uw2 = U[0][i-2][j];
        double vn1 = U[1][i][j+1];
        double vn2 = U[1][i][j+2];
        double vs1 = U[1][i][j-1];
        double vs2 = U[1][i][j-2];
        double dudx = (- ue2 + 8*ue1 - 8*uw1 + uw2)/(12*DX);
        double dvdy = (- vn2 + 8*vn1 - 8*vs1 + vs2)/(12*DY);
        RMS_DIV += sq(dudx + dvdy);
    }}
    RMS_DIV = sqrt(RMS_DIV/CXY);
}

void main_loop() {
    generate_force();
    prediction();
    periodic_bc(OMG, 2);
    sor_poisson(POIA, PSI, RHS, lsvar);
    psi_centralize();
    periodic_bc(PSI, 2);
    psi2u();
    periodic_bc(U[0], 2);
    periodic_bc(U[1], 2);
    calc_divergence();
    calc_turb_k();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, OMG, PSI, FF)
    char fname[128];
    sprintf(fname, "data/vorteq.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,omega,psi,f\n");
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., OMG[i][j], PSI[i][j], FF[i][j]);
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

void init_field() {
    #pragma acc parallel loop independent collapse(2) present(U)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        U[0][i][j] = UINFLOW;
        U[1][i][j] = VINFLOW;
    }}
}

int main() {
    NNX = int(LOW_PASS*LX/(2*PI)) + 1;
    NNY = int(LOW_PASS*LY/(2*PI)) + 1;
    printf("filtered wavenumber space %dx%d\n", NNX, NNY);

    ffk = (complex*)malloc(sizeof(complex)*NNX*NNY);

    make_grid();
    #pragma acc enter data copyin(ffk[0:NNX*NNY])
    init_env();
    lsvar.init();
    init_field();

    make_eq();
    printf("max diag = %lf\n", 1./MAXDIAGI);
    // generate_force();
    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        if (ISTEP%int(1./DT) == 0 && ISTEP >= int(OUPUT_START/DT)) {
            output_field(ISTEP/int(1./DT));
            printf("\n");
        }
        printf("\r%8d, %9.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_K_AVG, TURB_I, TURB_I_AVG, MAX_CFL);
        fflush(stdout);
    }
    printf("\n");

    #pragma acc exit data delete(ffk[0:NNX*NNY])
    finalize_env();
    lsvar.finalize();

    free(ffk);
    return 0;
}
