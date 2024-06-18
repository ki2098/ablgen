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
const static int GC = 2;
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
const static double HIGH_PASS = 7;
const static double FORCING_EFK = 1;
const static double DRAG = 1e-2*FORCING_EFK;

const static double UINFLOW = 0.5;
const static double VINFLOW = 0.;

const static int OUTPUT_OUTFLOW_OUTER = 1;
const static int OUTPUT_OUTFLOW_INNER = GC;

double MAX_CFL;

random_device RD;
default_random_engine GEN(RD());
normal_distribution<double> GAUSS(0., 1.);
uniform_real_distribution<double> UNI(0., 2*PI);

double X[CCX] = {};
double Y[CCY] = {};
double U[2][CCX][CCY] = {};
double OMEGA[CCX][CCY] = {};
double OMEGAP[CCX][CCY] = {};
double PSI[CCX][CCY] = {};
double PMAT[5][CCX][CCY] = {};
double RHS[CCX][CCY] = {};
double F[CCX][CCY];

const static int NNX = int(LOW_PASS*LX/(2*PI)) + 1;
const static int NNY = int(LOW_PASS*LY/(2*PI)) + 1;

complex FF[NNX][NNY] = {};

struct BoundaryOutputHandler {
    FILE *file;
    int output_start_x, output_size_x;
    int output_start_y, output_size_y;
    int output_snapshot_numbers;
    double mainstream_u, mainstream_v;

    void init(int startx, int sizex, int starty, int sizey, double mainu, double mainv, string fname) {
        file = fopen(fname.c_str(), "wb");
        output_snapshot_numbers = 0;
        output_start_x = startx;
        output_size_x = sizex;
        output_start_y = starty;
        output_size_y = sizey;
        mainstream_u = mainu;
        mainstream_v = mainv;
        int nvar = 2;
        fwrite(&output_size_x, sizeof(int), 1, file);
        fwrite(&output_size_y, sizeof(int), 1, file);
        fwrite(&nvar, sizeof(int), 1, file);
        fwrite(&output_snapshot_numbers, sizeof(int), 1, file);
        fwrite(&mainstream_u, sizeof(double), 1, file);
        fwrite(&mainstream_v, sizeof(double), 1, file);
    }

    void write(double u[2][CCX][CCY]) {
        for (int i = output_start_x; i < output_start_x + output_size_x; i ++) {
        for (int j = output_start_y; j < output_start_y + output_size_y; j ++) {
            fwrite(&u[0][i][j], sizeof(double), 1, file);
        }}
        for (int i = output_start_x; i < output_start_x + output_size_x; i ++) {
        for (int j = output_start_y; j < output_start_y + output_size_y; j ++) {
            fwrite(&u[1][i][j], sizeof(double), 1, file);
        }}
        output_snapshot_numbers ++;
    }

    void finalize() {
        fseek(file, sizeof(int)*3, SEEK_SET);
        fwrite(&output_snapshot_numbers, sizeof(int), 1, file);
        fclose(file);
        printf("boundary output=%dx%dx%dx%d, mainstream u=(%lf,%lf)\n", output_size_x, output_size_y, 2, output_snapshot_numbers, mainstream_u, mainstream_v);
    }
} boundaryOutput, initialOutput;

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

void periodic(double field[CCX][CCY], int margin) {
    #pragma acc parallel loop independent collapse(2) present(field) firstprivate(margin)
    for (int j = GC; j < GC+CY; j ++) {
    for (int offset = 0; offset < margin; offset ++) {
        field[GC- 1-offset][j] = field[GC+CX-1-offset][j];
        field[GC+CX+offset][j] = field[GC     +offset][j];
    }}
    #pragma acc parallel loop independent collapse(2) present(field) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int offset = 0; offset < margin; offset ++) {
        field[i][GC- 1-offset] = field[i][GC+CY-1-offset];
        field[i][GC+CY+offset] = field[i][GC     +offset];
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
    return REI*(dx2 + dy2);
}

void prediction() {
    copy_field(OMEGAP, OMEGA);
    #pragma acc parallel loop independent collapse(2) present(OMEGA, OMEGAP, U, F, RHS) firstprivate(MAXDIAGI)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double advc = advection_core(OMEGAP, U[0], U[1], i, j);
        double diff = diffusion_core(OMEGAP, i, j);
        OMEGA[i][j] = OMEGAP[i][j] + DT*(- advc + diff + F[i][j] - DRAG*OMEGAP[i][j]);
        RHS[i][j] = - OMEGA[i][j]*MAXDIAGI;
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
    if ((i + j)%2 == color) {
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
        double dx = DXI*(- psie2 + 8*psie1 - 8*psiw1 + psiw2)/12.;
        double dy = DYI*(- psin2 + 8*psin1 - 8*psis1 + psis2)/12.;
        U[0][i][j] =   dy + UINFLOW;
        U[1][i][j] = - dx + VINFLOW;
    }}
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

void kforce(complex f[NNX][NNY], int i, int j) {
    double th = UNI(GEN);
    f[i][j][REAL] = cos(th)*FORCING_EFK;
    f[i][j][IMAG] = sin(th)*FORCING_EFK;
    // f[i][j][REAL] = .2;
    // f[i][j][IMAG] = .2;
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
        F[i][j] = 0;
        int I1 = i - GC;
        int I2 = j - GC;
        for (int K1 = 0; K1 < NNX; K1 ++) {
        for (int K2 = 0; K2 < NNY; K2 ++) {
            double th1 = - 2.*PI*I1*K1/double(CX);
            double th2 = - 2.*PI*I2*K2/double(CY);
            double Real = cos(th1 + th2);
            double Imag = sin(th1 + th2);
            F[i][j] += FF[K1][K2][REAL]*Real - FF[K1][K2][IMAG]*Imag;
        }}
    }}
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
        double dudx = DXI*(- ue2 + 8*ue1 - 8*uw1 + uw2)/12.;
        double dvdy = DYI*(- vn2 + 8*vn1 - 8*vs1 + vs2)/12.;
        RMS_DIV += sq(dudx + dvdy);
    }}
    RMS_DIV = sqrt(RMS_DIV/CXY);
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

void main_loop() {
    generate_force();
    prediction();
    periodic(OMEGA, GC);

    // sor_poisson(PMAT, PSI, RHS, lsvar);
    pbicgstab_poisson(PMAT, PSI, RHS, lsvar);
    field_centralize(PSI);
    periodic(PSI, GC);

    psi2u();
    periodic(U[0], GC);
    periodic(U[1], GC);

    calc_divergence();
    calc_turb_k();
    max_cfl();
}

void output_field(int n) {
    #pragma acc update self(U, OMEGA, F, PSI)
    char fname[128];
    sprintf(fname, "data/vforcing.csv.%d", n);
    FILE *file = fopen(fname, "w");
    fprintf(file, "x,y,z,u,v,w,vorticity,psi,f\n");
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., OMEGA[i][j], PSI[i][j], F[i][j]);
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

void init_field() {
    #pragma acc parallel loop independent collapse(2) present(U)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        U[0][i][j] = UINFLOW;
        U[1][i][j] = VINFLOW;
    }}
}

int main() {
    make_grid();
    print_wavenumber_map();
    init_env();
    init_field();
    make_eq();
    printf("max diag=%lf\n", 1./MAXDIAGI);
    printf("Forcing coefficient=%lf\n", FORCING_EFK);
    printf("Drag coefficient=%lf\n", DRAG);

    boundaryOutput.init(0, 3, GC, CY, UINFLOW, VINFLOW, "data/inflow_boundary");
    initialOutput.init(0, CCX, 0, CCY, UINFLOW, VINFLOW, "data/initial_field");

    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%9d, %10.5lf, %3d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K, TURB_K_AVG, MAX_CFL);
        fflush(stdout);
        if (ISTEP%int(OUTPUT_INTERVAL/DT) == 0 && ISTEP >= int(OUTPUT_START/DT)) {
            output_field(ISTEP/int(OUTPUT_INTERVAL/DT));
            printf("\n");
        }
        if (ISTEP >= int(OUTPUT_START/DT)) {
            #pragma acc update self(U)
            boundaryOutput.write(U);
            if (ISTEP == int(OUTPUT_START/DT)) {
                initialOutput.write(U);
                initialOutput.finalize();
            }
        }
        if (ISTEP >= int(STATIC_AVG_START/DT)) {
            int nstep = ISTEP - int(STATIC_AVG_START/DT) + 1;
            TURB_K_AVG = ((nstep - 1)*TURB_K_AVG + TURB_K)/nstep;
        }
    }

    finalize_env();
    boundaryOutput.finalize();

    return 0;
}
