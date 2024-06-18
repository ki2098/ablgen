#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

typedef double complex[2];
const static int REAL = 0;
const static int IMAG = 1;

using namespace std;
const static double PI = M_PI;

const static int CX = 100;
const static int CY = 100;
const static int GC = 3;
const static int CXY = CX*CY;
const static int CCX = CX+2*GC;
const static int CCY = CY+2*GC;
const static int CCXY = CCX*CCY;

const static double LX = 5;
const static double LY = 5;
const static double DX = LX/CX;
const static double DY = LY/CY;
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

int ISTEP;
const static double MAXT        = 1000.;
const static double STATIC_AVG_START =500.;
const static double OUTPUT_START = 900.;
const static double OUTPUT_INTERVAL=1.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;

const static double C_SMAGORINSKY = 0.1;
double              TURB_K, TURB_K_AVG=0.;

const static double LOW_PASS = 10;
const static double HIGH_PASS = 8;
const static double FORCING_EFK = 0.5;
const static double DRAG = 6e-2*FORCING_EFK;

const static double UINFLOW = 0.;
const static double VINFLOW = 0.;

double MAX_CFL;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);
uniform_real_distribution<double> UNI(0., 2*PI);

double X[CCX] = {};
double Y[CCY] = {};
double U[2][CCX][CCY] = {};
double UU[2][CCX][CCY] = {};
double UP[2][CCX][CCY] = {};
double P[CCX][CCY] = {};
double RHS[CCX][CCY] = {};
double F[2][CCX][CCY] = {};
double PMAT[5][CCX][CCY] = {};

const static int NNX = int(LOW_PASS*LX/(2*PI)) + 1;
const static int NNY = int(LOW_PASS*LY/(2*PI)) + 1;

complex FF[2][NNX][NNY] = {};

void init_env() {
    #pragma acc enter data copyin(U, UU, UP, P, RHS, F, FF, PMAT)
}

void finalize_env() {
    #pragma acc exit data delete(U, UU, UP, P, RHS, F, FF, PMAT)
}

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

double gettime() {
    return ISTEP*DT;
}

template<typename T>
inline T sq(T a) {
    return a*a;
}

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

void periodic(double phi[CCX][CCY], int margin) {
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
    double uue = uu[i  ][j];
    double uuw = uu[i-1][j];
    double vvn = vv[i][j  ];
    double vvs = vv[i][j-1];
    double ucc = u[i][j];
    double vcc = v[i][j];
    double dx1e = (- phie2 + 27*phie1 - 27*phicc + phiw1)*DXI*uue;
    double dx1w = (- phie1 + 27*phicc - 27*phiw1 + phiw2)*DXI*uuw;
    double dy1n = (- phin2 + 27*phin1 - 27*phicc + phis1)*DYI*vvn;
    double dy1s = (- phin1 + 27*phicc - 27*phis1 + phis2)*DYI*vvs;
    double dx4 = (phie2 -4*phie1 + 6*phicc - 4*phiw1 + phiw2)*DXI*fabs(ucc);
    double dy4 = (phin2 -4*phin1 + 6*phicc - 4*phis1 + phis2)*DYI*fabs(vcc);
    return (.5*(dx1e + dx1w + dy1n + dy1s) + (dx4 + dy4))/24;
}

double diffusion_core(double phi[CCX][CCY], int i, int j) {
    double phic = phi[i][j];
    double phie = phi[i+1][j];
    double phiw = phi[i-1][j];
    double phin = phi[i][j+1];
    double phis = phi[i][j-1];
    return REI*(DDXI*(phie - 2*phic + phiw) + DDYI*(phin - 2*phic + phis));
}

void prediction() {
    #pragma acc parallel loop independent collapse(2) present(U, UP, UU, F)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int d = 0; d < 2; d ++) {
        double advc = advection_core(UP[d], UP[0], UP[1], UU[0], UU[1], i, j);
        double diff = diffusion_core(UP[d], i, j);
        double drag = DRAG*UP[d][i][j];
        U[d][i][j] = UP[d][i][j] + DT*(- advc + diff - drag + F[d][i][j]);
    }}}
}

void interpolation(double max_diag_inverse) {
    #pragma acc parallel loop independent collapse(2) present(U, UU)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] = .5*(U[0][i][j] + U[0][i+1][j]);
    }}
    #pragma acc parallel loop independent collapse(2) present(U, UU)
    for (int i = GC  ; i < GC+CY; i ++) {
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

double calc_norm2sq(double vec[CCX][CCY]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(vec)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += sq(vec[i][j]);
    }}
    return sum;
}

void calc_res(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], double r[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(a, x, b, r)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double ac = a[0][i][j];
        double ae = a[1][i][j];
        double aw = a[2][i][j];
        double an = a[3][i][j];
        double as = a[4][i][j];
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        double xc = x[i][j];
        double xe = x[ie][j];
        double xw = x[iw][j];
        double xn = x[i][jn];
        double xs = x[i][js];
        r[i][j] = b[i][j] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs);
    }}
}

void calc_ax(double a[5][CCX][CCY], double x[CCX][CCY], double ax[CCX][CCY]) {
    #pragma acc parallel loop independent collapse(2) present(a, x, ax)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double ac = a[0][i][j];
        double ae = a[1][i][j];
        double aw = a[2][i][j];
        double an = a[3][i][j];
        double as = a[4][i][j];
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        double xc = x[i][j];
        double xe = x[ie][j];
        double xw = x[iw][j];
        double xn = x[i][jn];
        double xs = x[i][js];
        ax[i][j] = ac*xc + ae*xe + aw*xw + an*xn + as*xs;
    }}
}

double rbsor_core(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int i, int j, int color) {
    if ((i + j) % 2 == color) {
        double ac = a[0][i][j];
        double ae = a[1][i][j];
        double aw = a[2][i][j];
        double an = a[3][i][j];
        double as = a[4][i][j];
        int ie = (i < GC+CX-1)? i+1 : GC     ;
        int iw = (i > GC     )? i-1 : GC+CX-1;
        int jn = (j < GC+CY-1)? j+1 : GC     ;
        int js = (j > GC     )? j-1 : GC+CY-1;
        double xc = x[i][j];
        double xe = x[ie][j];
        double xw = x[iw][j];
        double xn = x[i][jn];
        double xs = x[i][js];
        double cc = (b[i][j] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs))/ac;
        x[i][j] = xc + SOR_OMEGA*cc;
        return cc*cc;
    } else {
        return 0;
    }
}

void sor_poisson(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], LSVAR &var) {
    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 0);
        }}
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 1);
        }}
        calc_res(a, x, b, var.r);
        LS_ERR = sqrt(calc_norm2sq(var.r)/CXY);
        if (LS_ERR < LS_EPS) {
            break;
        }
    }
}

void sor_preconditioner(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int maxit) {
    for (int iter = 1; iter <= maxit; iter ++) {
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 0);
        }}
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 1);
        }}
    }
}

void field_centralize(double field[CCX][CCY]) {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(field)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += field[i][j];
    }}
    double avg = sum/double(CXY);
    #pragma acc parallel loop independent collapse(2) present(field) firstprivate(avg)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        field[i][j] -= avg;
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

void pbicgstab_poisson(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], LSVAR &var) {
    calc_res(a, x, b, var.r);
    LS_ERR = sqrt(calc_norm2sq(var.r)/CXY);
    copy_field(var.r0, var.r);

    double rho, rrho=1., alpha=1., beta, omega=1.;
    set_field(var.q, 0);

    for (LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        rho = dot_product(var.r, var.r0);
        if (fabs(rho) < __FLT_MIN__) {
            LS_ERR = fabs(rho);
            break;
        }

        if (LS_ERR == 1) {
            copy_field(var.p, var.r);
        } else {
            beta = (rho*alpha)/(rrho*omega);
            #pragma acc parallel loop independent collapse(2) present(var, var.p, var.r, var.q) firstprivate(beta, omega)
            for (int i = GC; i < GC+CX; i ++) {
            for (int j = GC; j < GC+CY; j ++) {
                var.p[i][j] = var.r[i][j] + beta*(var.p[i][j] - omega*var.q[i][j]);
            }}
        }

        set_field(var.pp, 0);
        sor_preconditioner(a, var.pp, var.p, 3);
        calc_ax(a, var.pp, var.q);

        alpha = rho/dot_product(var.r0, var.q);

        #pragma acc parallel loop independent collapse(2) present(var, var.s, var.r, var.q) firstprivate(alpha)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            var.s[i][j] = var.r[i][j] - alpha*var.q[i][j];
        }}

        set_field(var.ss, 0);
        sor_preconditioner(a, var.ss, var.s, 3);
        calc_ax(a, var.ss, var.t);

        omega = dot_product(var.t, var.s)/dot_product(var.t, var.t);

        #pragma acc parallel loop independent collapse(2) present(x, var, var.pp, var.ss) firstprivate(alpha, omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            x[i][j] += alpha*var.pp[i][j] + omega*var.ss[i][j];
        }}

        #pragma acc parallel loop independent collapse(2) present(var, var.r, var.s, var.t) firstprivate(omega)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            var.r[i][j] = var.s[i][j] - omega*var.t[i][j];
        }}

        rrho = rho;
        LS_ERR = sqrt(calc_norm2sq(var.r)/CXY);
        if (LS_ERR < LS_EPS) {
            break;
        }
    }
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
    for (int i = GC  ; i < GC+CY; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[1][i][j] -= DT*DYI*(P[i][j+1] - P[i][j]);
    }}
}

void calc_divergence() {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(UU)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double div = DXI*(UU[0][i][j] - UU[0][i-1][j]);
        div       += DYI*(UU[1][i][j] - UU[1][i][j-1]);
        sum += div*div;
    }}
    RMS_DIV = sqrt(sum/CXY);
}

void max_cfl() {
    MAX_CFL = 0;
    #pragma acc parallel loop independent reduction(max:MAX_CFL) collapse(2) present(U)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double uabs = fabs(U[0][i][j]);
        double vabs = fabs(U[1][i][j]);
        double local_cfl = DT*(uabs/DX + vabs/DY);
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
        double u = U[0][i][j];
        double v = U[1][i][j];
        TURB_K += .5*sqrt(u*u + v*v);
    }}
    TURB_K /= CXY;
}

void kforce(complex f[2][NNX][NNY], int i, int j) {
    
        // double a1, a2, a3, b1, b2, b3, k1, k2, k3;
        // k1 = i*2*PI/LX;
        // k2 = j*2*PI/LY;
        // k3 = 0;
        // a1 = GAUSS(GEN);
        // a2 = GAUSS(GEN);
        // a3 = GAUSS(GEN);
        // b1 = GAUSS(GEN);
        // b2 = GAUSS(GEN);
        // b3 = GAUSS(GEN);
        // double kabs = sqrt(k1*k1 + k2*k2 + k3*k3);
        // double Cf = sqrt(FORCING_EFK/(16*PI*sq(sq(kabs))*DT));
        // f[0][i][j][REAL] = Cf*(k2*a3 - k3*a2);
        // f[0][i][j][IMAG] = Cf*(k2*b3 - k3*b2);
        // f[1][i][j][REAL] = Cf*(k3*a1 - k1*a3);
        // f[1][i][j][IMAG] = Cf*(k3*b1 - k1*b3);

    double th = UNI(GEN);
    f[0][i][j][REAL] = cos(th)*FORCING_EFK;
    f[0][i][j][IMAG] = sin(th)*FORCING_EFK;
    th = UNI(GEN);
    f[1][i][j][REAL] = cos(th)*FORCING_EFK;
    f[1][i][j][IMAG] = sin(th)*FORCING_EFK;
        
    // f[0][i][j][REAL] = 0.177;
    // f[1][i][j][REAL] = 0.177;
        
    
}

bool wavenumber_pass(int i, int j) {
    double k1 = i*2*PI/LX;
    double k2 = j*2*PI/LY;
    double kabs = sqrt(k1*k1 + k2*k2);
    if (i + j == 0) {
        return false;
    } else if (kabs >= HIGH_PASS && kabs <= LOW_PASS) {
        return true;
    } else {
        return false;
    }
}

void generate_force() {
    for (int i = 0; i < NNX; i ++) {
    for (int j = 0; j < NNY; j ++) {
        if (wavenumber_pass(i,j)) {
            kforce(FF, i, j);
        }
    }}
    #pragma acc update device(FF)
    #pragma acc parallel loop independent collapse(2) present(F, FF)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        F[0][i][j] = 0.;
        F[1][i][j] = 0.;
        int I1 = i - GC;
        int I2 = j - GC;
        for (int K1 = 0; K1 < NNX; K1 ++) {
        for (int K2 = 0; K2 < NNY; K2 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double Real = cos(th1 + th2);
            double Imag = sin(th1 + th2);
            F[0][i][j] += FF[0][K1][K2][REAL]*Real - FF[0][K1][K2][IMAG]*Imag;
            F[1][i][j] += FF[1][K1][K2][REAL]*Real - FF[1][K1][K2][IMAG]*Imag;
        }}
    }}
}

void main_loop() {
    copy_field(UP[0], U[0]);
    copy_field(UP[1], U[1]);
    generate_force();
    prediction();
    periodic(U[0], 2);
    periodic(U[1], 2);
    interpolation(MAXDIAGI);

    // sor_poisson(PMAT, P, RHS, lsvar);
    pbicgstab_poisson(PMAT, P, RHS, lsvar);
    field_centralize(P);
    periodic(P, 1);

    projection_center();
    projection_interface();
    periodic(U[0], 2);
    periodic(U[1], 2);

    calc_divergence();
    calc_turb_k();
    max_cfl();
}

void make_grid() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC + .5)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC + .5)*DY;
    }
}

void make_eq() {
    double max_diag = 0;
    #pragma acc parallel loop independent reduction(max:max_diag) collapse(2) present(PMAT)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double ae = DDXI;
        double aw = DDXI;
        double an = DDYI;
        double as = DDYI;
        double ac = - (ae + aw + an + as);
        PMAT[0][i][j] = ac;
        PMAT[1][i][j] = ae;
        PMAT[2][i][j] = aw;
        PMAT[3][i][j] = an;
        PMAT[4][i][j] = as;
        if (fabs(ac) > max_diag) {
            max_diag = fabs(ac);
        }
    }}
    MAXDIAGI = 1./max_diag;
    #pragma acc parallel loop independent collapse(2) present(PMAT) firstprivate(MAXDIAGI)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        PMAT[0][i][j] *= MAXDIAGI;
        PMAT[1][i][j] *= MAXDIAGI;
        PMAT[2][i][j] *= MAXDIAGI;
        PMAT[3][i][j] *= MAXDIAGI;
        PMAT[4][i][j] *= MAXDIAGI;
    }}
}

void print_wavenumber_map() {
    for (int j = NNY - 1; j >= 0; j --) {
    for (int i = 0; i < NNX; i ++) {
        if (wavenumber_pass(i,j)) {
            printf("+ ");
        } else {
            printf(". ");
        }
    }
    printf("\n");
    }
}

void output_field(int n) {
    #pragma acc update self(U, P, F)
    char fname[128];
    sprintf(fname, "data/fforcing.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,p,f1,f2,f3\n");
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., P[i][j], F[0][i][j], F[1][i][j], 0.);
    }}
    fclose(file);
}

int main() {
    make_grid();
    print_wavenumber_map();
    init_env();
    lsvar.init();
    make_eq();
    printf("max diag=%lf\n", 1./MAXDIAGI);
    printf("Forcing coefficient=%lf\n", FORCING_EFK);
    printf("Drag coefficient=%lf\n", DRAG);

    FILE *statistics_file = fopen("data/statistics.csv", "w");
    fprintf(statistics_file, "t,k,kavg\n");

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%9d, %10.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_K_AVG, MAX_CFL);
        fflush(stdout);
        if (ISTEP%int(OUTPUT_INTERVAL/DT) == 0 && ISTEP >= int(OUTPUT_START/DT)) {
            output_field(ISTEP/int(OUTPUT_INTERVAL/DT));
            printf("\n");
        }
        if (ISTEP >= int(STATIC_AVG_START/DT)) {
            int nstep = ISTEP - int(STATIC_AVG_START/DT) + 1;
            TURB_K_AVG = ((nstep - 1)*TURB_K_AVG + TURB_K)/nstep;
            if (ISTEP%int(OUTPUT_INTERVAL/DT) == 0) {
                fprintf(statistics_file, "%12.5e,%12.5e,%12.5e\n", gettime(), TURB_K, TURB_K_AVG);
            }
        }
    }
    printf("\n");
    fclose(statistics_file);

    lsvar.finalize();
    finalize_env();
    return 0;
}
