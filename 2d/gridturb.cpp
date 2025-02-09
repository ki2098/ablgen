#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>

using namespace std;
const static double PI = M_PI;

const static int DivisionX = 500;
const static int DivisionY = 200;
const static int CX = DivisionX;
const static int CY = DivisionY-1;
const static int GC = 2;
const static int CXY = CX*CY;
const static int CCX = CX+2*GC;
const static int CCY = CY+2*GC;
const static int CCXY = CCX*CCY;

const static double LX = 5;
const static double LY = 2;
const static double DX = LX/DivisionX;
const static double DY = LY/DivisionY;
const static double DXI = 1./DX;
const static double DYI = 1./DY;
const static double DDXI = DXI*DXI;
const static double DDYI = DYI*DYI;
const static double DT = 2e-4;
const static double DTI = 1./DT;

const static double RE = 1e4;
const static double REI = 1./RE;

const static double SOR_OMEGA = 1.2;
int LS_ITER;
const static int LS_MAXITER = 1000;
const static double LS_EPS = 1e-3;
double LS_ERR;

int ISTEP;
const static double MAXT        = 50.;
const static double OUTPUT_START = 25.;
const static double OUTPUT_INTERVAL=1.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;
double              MAX_CFL;

const static double C_SMAGORINSKY = 0.1;

const static int    GRID_NUM = 21;
const static double GRID_THICKNESS_X = 0.05;
const static double GRID_THICKNESS_Y = 0.0205;
double GRID_X[GRID_NUM];
double GRID_Y[GRID_NUM];

double P_DRIVER = 0;
double P_DRIVER_COEFFICIENT = 1.;

double gettime() {
    return ISTEP*DT;
}

template<typename T>
inline T sq(T a) {
    return a*a;
}

void periodic_x(double field[CCX][CCY], int margin, double df) {
    #pragma acc parallel loop independent collapse(2) present(field) firstprivate(margin, df)
    for (int j = 0; j < CCY; j ++) {
    for (int offset = 0; offset < margin; offset ++) {
        field[GC- 1-offset][j] = field[GC+CX-1-offset][j] - df;
        field[GC+CX+offset][j] = field[GC     +offset][j] + df;
    }}
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

double X[CCX] = {};
double Y[CCY] = {};
double U[2][CCX][CCY] = {};
double UU[2][CCX][CCY] = {};
double UP[2][CCX][CCY] = {};
double P[CCX][CCY] = {};
double RHS[CCX][CCY] = {};
double PMAT[5][CCX][CCY] = {};
double DIV[CCX][CCY] = {};
double NUT[CCX][CCY] = {};
int FLAG[CCX][CCY] = {};

void acc_init() {
    #pragma acc enter data create(X, Y, U, UU, UP, P, RHS, PMAT, DIV, NUT, FLAG, GRID_X, GRID_Y)
}

void acc_finalize() {
    #pragma acc exit data delete(X, Y, U, UU, UP, P, RHS, PMAT, DIV, NUT, FLAG, GRID_X, GRID_Y)
}

double outflow_monitor() {
    double usum = 0;
    #pragma acc parallel loop independent reduction(+:usum) present(U, FLAG)
    for (int j = GC; j < GC+CY; j ++) {
        usum += U[0][GC+CX-1][j];
    }
    return usum/CY;
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

double diffusion_core(double phi[CCX][CCY], double nut[CCX][CCY], int i, int j) {
    double phic = phi[i][j];
    double phie = phi[i+1][j];
    double phiw = phi[i-1][j];
    double phin = phi[i][j+1];
    double phis = phi[i][j-1];
    double nute = .5*(nut[i+1][j] + nut[i][j]);
    double nutw = .5*(nut[i-1][j] + nut[i][j]);
    double nutn = .5*(nut[i][j+1] + nut[i][j]);
    double nuts = .5*(nut[i][j-1] + nut[i][j]);
    double dife = (REI + nute)*DXI*(phie - phic);
    double difw = (REI + nutw)*DXI*(phic - phiw);
    double difn = (REI + nutn)*DYI*(phin - phic);
    double difs = (REI + nuts)*DYI*(phic - phis);
    return DXI*(dife - difw) + DYI*(difn - difs);
}

void prediction() {
    #pragma acc parallel loop independent collapse(2) present(U, UP, UU, NUT)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int d = 0; d < 2; d ++) {
        double advc = advection_core(UP[d], UP[0], UP[1], UU[0], UU[1], i, j);
        double diff = diffusion_core(UP[d], NUT, i, j);
        U[d][i][j] = UP[d][i][j] + DT*(- advc + diff);
    }}}
}

void interpolation(double max_diag_inverse) {
    #pragma acc parallel loop independent collapse(2) present(U, UU)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[0][i][j] = .5*(U[0][i][j] + U[0][i+1][j]);
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
        int ie = i+1;
        int iw = i-1;
        int jn = j+1;
        int js = j-1;
        double xc = x[i][j];
        double xe = x[ie][j];
        double xw = x[iw][j];
        double xn = x[i][jn];
        double xs = x[i][js];
        r[i][j] = b[i][j] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs);
    }}
}

double rbsor_core(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int i, int j, int color) {
    if ((i + j) % 2 == color) {
        double ac = a[0][i][j];
        double ae = a[1][i][j];
        double aw = a[2][i][j];
        double an = a[3][i][j];
        double as = a[4][i][j];
        int ie = i+1;
        int iw = i-1;
        int jn = j+1;
        int js = j-1;
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
        periodic_x(P, GC, P_DRIVER);
        #pragma acc parallel loop independent collapse(2) present(a, x, b)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            rbsor_core(a, x, b, i, j, 1);
        }}
        periodic_x(P, GC, P_DRIVER);
        calc_res(a, x, b, var.r);
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
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[0][i][j] -= DT*DXI*(P[i+1][j] - P[i][j]);
        UU[1][i][j] -= DT*DYI*(P[i][j+1] - P[i][j]);
    }}
}

void calc_divergence() {
    double sum = 0;
    #pragma acc parallel loop independent reduction(+:sum) collapse(2) present(UU, DIV)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double div = DXI*(UU[0][i][j] - UU[0][i-1][j]);
        div       += DYI*(UU[1][i][j] - UU[1][i][j-1]);
        sum += div*div;
        DIV[i][j] = div;
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

void velocity_bc() {
    #pragma acc parallel loop independent present(U, FLAG)
    for (int i = GC; i < GC+CX; i ++) {
        if (!FLAG[i][GC-1]) {
            U[0][i][GC-1] = U[0][i][GC  ];
            U[1][i][GC-1] = 0;
        }
        U[0][i][GC-2] = 2*U[0][i][GC-1] - U[0][i][GC];
        U[1][i][GC-2] = 2*U[1][i][GC-1] - U[1][i][GC];

        if (!FLAG[i][GC+CY]) {
            U[0][i][GC+CY  ] = U[0][i][GC+CY-1];
            U[1][i][GC+CY  ] = 0;
        }
        U[0][i][GC+CY+1] = 2*U[0][i][GC+CY  ] - U[0][i][GC+CY-1];
        U[1][i][GC+CY+1] = 2*U[1][i][GC+CY  ] - U[1][i][GC+CY-1];
    }
    periodic_x(U[0], GC, 0);
    periodic_x(U[1], GC, 0);
}

void pressure_bc() {
    #pragma acc parallel loop independent present(P)
    for (int i = GC; i < GC+CX; i ++) {
        P[i][GC-1] = P[i][GC];
        P[i][GC+CY] = P[i][GC+CY-1];
    }
    periodic_x(P, GC, - P_DRIVER);
}

void turbulence_core(double u[2][CCX][CCY], double nut[CCX][CCY], int i, int j) {
    double ue = u[0][i+1][j];
    double uw = u[0][i-1][j];
    double un = u[0][i][j+1];
    double us = u[0][i][j-1];
    double ve = u[1][i+1][j];
    double vw = u[1][i-1][j];
    double vn = u[1][i][j+1];
    double vs = u[1][i][j-1];
    double dudx = .5*DXI*(ue - uw);
    double dudy = .5*DYI*(un - us);
    double dvdx = .5*DXI*(ve - vw);
    double dvdy = .5*DYI*(vn - vs);
    double Du = sqrt(2*dudx*dudx + 2*dvdy*dvdy + sq(dudy + dvdx));
    double De = sqrt(DX*DY);
    double LC = C_SMAGORINSKY*De;
    nut[i][j] = sq(LC)*Du;
}

void turbulence() {
    #pragma acc parallel loop independent collapse(2) present(U, NUT)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        turbulence_core(U, NUT, i, j);
    }}
}

void nut_bc() {
    #pragma acc parallel loop independent present(NUT)
    for (int i = GC; i < GC+CX; i ++) {
        NUT[i][GC- 1] = NUT[i][GC     ];
        NUT[i][GC+CY] = NUT[i][GC+CY-1];
    }
    periodic_x(NUT, GC, 0);
}

void grid_init() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC+1)*DY;
    }
    #pragma acc update device(X, Y)
}

void eq_init() {
    double max_diag = 0;
    #pragma acc parallel loop independent reduction(max:max_diag) collapse(2) present(PMAT)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double ae = DDXI;
        double aw = DDXI;
        double an = DDYI;
        // if (j == GC+CY-1) an = 0;
        double as = DDYI;
        // if (j == GC     ) as = 0;
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

#define U_INFLOW 1
#define V_INFLOW 0

void field_init() {
    #pragma acc parallel loop independent collapse(2) present(U, P)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        U[0][i][j] = U_INFLOW;
        U[1][i][j] = V_INFLOW;
        P[i][j] = 0;
    }}
}

void force_grid() {
    #pragma acc parallel loop independent collapse(2) present(U, FLAG, GRID_X, GRID_Y, X, Y)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        if (FLAG[i][j]) {
            U[0][i][j] = 0;
            U[1][i][j] = 0;
        }
    }}
}

void init() {
    acc_init();
    lsvar.init();
    grid_init();
    field_init();
    eq_init();

    for (int n = 0; n < GRID_NUM; n ++) {
        GRID_X[n] = 1;
        GRID_Y[n] = LY/(GRID_NUM-1)*n;
    }
    #pragma acc update device(GRID_X, GRID_Y)

    int true_fluid = 0;
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        for (int n = 0; n < GRID_NUM; n ++) {
            double dx = X[i] - GRID_X[n];
            double dy = Y[j] - GRID_Y[n];
            if (dx <= GRID_THICKNESS_X && dx >= 0 && fabs(dy) <= GRID_THICKNESS_Y*0.5) {
                FLAG[i][j] = 1;
                break;
            }
        }
    }}
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        if (!FLAG[i][j]) {
            true_fluid ++;
        }
    }}
    #pragma acc update device(FLAG)
    P_DRIVER_COEFFICIENT = CXY/double(true_fluid);
    printf("%e\n", P_DRIVER_COEFFICIENT);

    force_grid();
    velocity_bc();
    
    interpolation(MAXDIAGI);
    printf("max diag = %lf\n", 1./MAXDIAGI);
}

void finalize() {
    lsvar.finalize();
    acc_finalize();
}

void output_field(int n) {
    #pragma acc update self(U, P, DIV)
    string fname_base = "data/grid-turbulence.csv";
    vector<char> fname(fname_base.size() + 32);
    sprintf(fname.data(), "%s.%d", fname_base.c_str(), n);
    FILE *file = fopen(fname.data(), "w");
    fprintf(file, "x,y,z,u,v,w,p,div\n");
    for (int j = 0; j < CCY; j ++) {
    for (int i = 0; i < CCX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., P[i][j], DIV[i][j]);
    }}
    fclose(file);
}

void main_loop() {
    copy_field(UP[0], U[0]);
    copy_field(UP[1], U[1]);
    prediction();
    periodic_x(U[0], GC, 0);
    periodic_x(U[1], GC, 0);
    interpolation(MAXDIAGI);

    sor_poisson(PMAT, P, RHS, lsvar);
    field_centralize(P);
    pressure_bc();

    projection_center();
    projection_interface();
    force_grid();
    velocity_bc();

    // turbulence();
    // nut_bc();

    calc_divergence();
    max_cfl();

    double bulk_u = outflow_monitor();
    P_DRIVER = (U_INFLOW - bulk_u)*(CX*DX)*P_DRIVER_COEFFICIENT*2;
}

int main() {
    init();
    
    for (ISTEP = 0; ISTEP <= MAXSTEP; ISTEP ++) {
        if (ISTEP >= 1) {
            main_loop();
            printf("\r%9d, %10.5lf, %4d, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, MAX_CFL, outflow_monitor(), P_DRIVER);
            fflush(stdout);
        }
        if (ISTEP >= int(OUTPUT_START/DT) && ISTEP%int(OUTPUT_INTERVAL/DT) == 0) {
            output_field(ISTEP/int(OUTPUT_INTERVAL/DT));
            printf("\n");
        }
    }

    finalize();
    return 0;
}