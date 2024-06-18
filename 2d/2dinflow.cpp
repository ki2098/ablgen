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

const static int DivisionX = 400;
const static int DivisionY = 200;
const static int CX = DivisionX-1;
const static int CY = DivisionY;
const static int GC = 2;
const static int CXY = CX*CY;
const static int CCX = CX+2*GC;
const static int CCY = CY+2*GC;
const static int CCXY = CCX*CCY;

const static double LX = 10;
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

int ISTEP;
const static double MAXT        = 100.;
const static double OUTPUT_START = 0.;
const static double OUTPUT_INTERVAL=1.;
const static int    MAXSTEP     = int(MAXT/DT);
double              RMS_DIV;
double              MAXDIAGI = 1.;
double              MAX_CFL;

const static double C_SMAGORINSKY = 0.1;

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

struct FileHandler {
    int fxn, fyn, fvn, fsn;
    int xo, yo, xn, yn, vn=2;
    double *data;
    FILE *file;
    double mainstream_u, mainstream_v;

    int id(int n, int i, int j) {
        return n*fxn*fyn + i*fyn + j;
    }

    void init(int _xo, int _yo, int _xn, int _yn, string fname) {
        xo = _xo;
        yo = _yo;
        xn = _xn;
        yn = _yn;
        file = fopen(fname.c_str(), "rb");
        fread(&fxn, sizeof(int), 1, file);
        fread(&fyn, sizeof(int), 1, file);
        fread(&fvn, sizeof(int), 1, file);
        fread(&fsn, sizeof(int), 1, file);
        fread(&mainstream_u, sizeof(double), 1, file);
        fread(&mainstream_v, sizeof(double), 1, file);
        printf("%s: (%d %d %d %d) (%lf %lf)\n", fname.c_str(), fxn, fyn, fvn, fsn, mainstream_u, mainstream_v);
        data = (double*)malloc(sizeof(double)*fvn*fxn*fyn);
        #pragma acc enter data copyin(this[0:1]) create(data[0:fvn*fxn*fyn])
    }
    
    void finalize() {
        #pragma acc exit data delete(data[0:fvn*fxn*fyn], this[0:1])
        free(data);
        fclose(file);
    }

    void apply(int step, double u[2][CCX][CCY]) {
        size_t skip = 4*sizeof(int) + 2*sizeof(double) + sizeof(double)*fvn*fxn*fyn*step;
        fseek(file, skip, SEEK_SET);
        fread(data, sizeof(double), fvn*fxn*fyn, file);
        #pragma acc update device(data[0:fvn*fxn*fyn])
        #pragma acc parallel loop independent collapse(2) present(this[0:1], u, data[0:fvn*fxn*fyn])
        for (int i = 0; i < xn; i ++) {
        for (int j = 0; j < yn; j ++) {
            u[0][i+xo][j+yo] = data[id(0,i,j)];
            u[1][i+xo][j+yo] = data[id(1,i,j)];
        }}
    }
} inflowHandler, initialHandler;

// struct InflowHandler {
//     int imax, jmax, slicemax;
//     FILE *file;
//     double *inflow;
//     double mainstream_u, mainstream_v;

//     int id(int n, int i, int j) {
//         return n*imax*jmax + i*jmax + j;
//     }

//     void init(string fname, double u[2][CCX][CCY]) {
//         file = fopen(fname.c_str(), "rb");
//         fread(&imax, sizeof(int), 1, file);
//         fread(&jmax, sizeof(int), 1, file);
//         int nvar;
//         fread(&nvar, sizeof(int), 1, file);
//         fread(&slicemax, sizeof(int), 1, file);
//         if (imax != GC + 1) {
//             printf("!Wrong thickness of inflow layer: %d, it should be %d!\n", imax, GC + 1);
//         }
//         if (nvar != 2) {
//             printf("!Wrong number of variables in inflow file!\n");
//         }
//         if (slicemax != int(MAXT/DT) + 1) {
//             printf("!Wrong number of snapshots in inflow file\n");
//         }
//         fread(&mainstream_u, sizeof(double), 1, file);
//         fread(&mainstream_v, sizeof(double), 1, file);

//         printf("inflow: %dx%dx%dx%d (%lf,%lf)\n", imax, jmax, nvar, slicemax, mainstream_u, mainstream_v);
        
//         inflow = (double*)malloc(sizeof(double)*imax*jmax*2);
//         fread(inflow, sizeof(double), imax*jmax*2, file);
//         #pragma acc enter data copyin(this[0:1], inflow[0:2*imax*jmax])

//         #pragma acc parallel loop independent collapse(2) present(u, this[0:1])
//         for (int i = 0; i < CCX; i ++) {
//         for (int j = 0; j < CCY; j ++) {
//             u[0][i][j] = mainstream_u;
//             u[1][i][j] = mainstream_v;
//         }}
//         insert_velocity(u);
//     }

//     void finalize() {
//         #pragma acc exit data delete(inflow[0:2*imax*jmax], this[0:1])
//         free(inflow);
//         fclose(file);
//     }

//     void read(double u[2][CCX][CCY]) {
//         fread(inflow, sizeof(double), imax*jmax*2, file);
//         #pragma acc update device(inflow[0:2*imax*jmax])
//         insert_velocity(u);
//     }

//     void insert_velocity(double u[2][CCX][CCY]) {
//         #pragma acc parallel loop independent collapse(2) present(this[0:1], inflow[:2*imax*jmax], u)
//         for (int i = 0; i < GC; i ++) {
//         for (int j = 0; j < jmax; j ++) {
//             u[0][i][j+GC] = inflow[id(0,i,j)];
//             u[1][i][j+GC] = inflow[id(1,i,j)];
//         }}
//     }

// } inflow_handler;

double X[CCX] = {};
double Y[CCY] = {};
double U[2][CCX][CCY] = {};
double UU[2][CCX][CCY] = {};
double UP[2][CCX][CCY] = {};
double P[CCX][CCY] = {};
double RHS[CCX][CCY] = {};
double PMAT[5][CCX][CCY] = {};
double DIV[CCX][CCY] = {};

void acc_init() {
    #pragma acc enter data create(X, Y, U, UU, UP, P, RHS, PMAT, DIV)
}

void acc_finalize() {
    #pragma acc exit data delete(X, Y, U, UU, UP, P, RHS, PMAT, DIV)
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

void grid_init() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC + 1)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC)*DY;
    }
    #pragma acc update device(X, Y)
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
    #pragma acc parallel loop independent collapse(2) present(U, UP, UU)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
    for (int d = 0; d < 2; d ++) {
        double advc = advection_core(UP[d], UP[0], UP[1], UU[0], UU[1], i, j);
        double diff = diffusion_core(UP[d], i, j);
        U[d][i][j] = UP[d][i][j] + DT*(- advc + diff);
    }}}
}

void interpolation(double max_diag_inverse) {
    #pragma acc parallel loop independent collapse(2) present(U, UU)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] = .5*(U[0][i][j] + U[0][i+1][j]);
    }}
    #pragma acc parallel loop independent collapse(2) present(U, UU)
    for (int i = GC  ; i < GC+CX; i ++) {
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
        int ie = i+1;
        int iw = i-1;
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

double rbsor_core(double a[5][CCX][CCY], double x[CCX][CCY], double b[CCX][CCY], int i, int j, int color) {
    if ((i + j) % 2 == color) {
        double ac = a[0][i][j];
        double ae = a[1][i][j];
        double aw = a[2][i][j];
        double an = a[3][i][j];
        double as = a[4][i][j];
        int ie = i+1;
        int iw = i-1;
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
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] -= DT*DXI*(P[i+1][j] - P[i][j]);
    }}
    #pragma acc parallel loop independent collapse(2) present(UU, P)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
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

void periodic_y(double phi[CCX][CCY], int margin) {
    #pragma acc parallel loop independent collapse(2) present(phi) firstprivate(margin)
    for (int i = GC; i < GC+CX; i ++) {
    for (int offset = 0; offset < margin; offset ++) {
        phi[i][GC- 1-offset] = phi[i][GC+CY-1-offset];
        phi[i][GC+CY+offset] = phi[i][GC     +offset];
    }}
}

void velocity_bc() {
    inflowHandler.apply(ISTEP, U);
    periodic_y(U[0], GC);
    periodic_y(U[1], GC);
    #pragma acc parallel loop independent present(U, UP, inflowHandler)
    for (int j = GC; j < GC+CY; j ++) {
        int icc = GC+CX;
        int ii1 = icc-1;
        int ii2 = icc-2;
        int io1 = icc+1;

        double dudx = .5*DXI*(3*UP[0][icc][j] - 4*UP[0][ii1][j] + UP[0][ii2][j]);
        double dvdx = .5*DXI*(3*UP[1][icc][j] - 4*UP[1][ii1][j] + UP[1][ii2][j]);
        U[0][icc][j] = UP[0][icc][j] - inflowHandler.mainstream_u*DT*dudx;
        U[1][icc][j] = UP[1][icc][j] - inflowHandler.mainstream_u*DT*dvdx;

        dudx        = .5*DXI*(3*UP[0][io1][j] - 4*UP[0][icc][j] + UP[0][ii1][j]);
        dvdx        = .5*DXI*(3*UP[1][io1][j] - 4*UP[1][icc][j] + UP[1][ii1][j]);
        U[0][io1][j] = UP[0][io1][j] - inflowHandler.mainstream_u*DT*dudx;
        U[1][io1][j] = UP[1][io1][j] - inflowHandler.mainstream_u*DT*dvdx;
    }
}

void pressure_bc() {
    periodic_y(P, 1);
    #pragma acc parallel loop independent present(P)
    for (int j = GC; j < GC+CY; j ++) {
        P[GC- 1][j] = P[GC     ][j];
        P[GC+CX][j] = P[GC+CX-1][j];
    }
}

void eq_init() {
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

void field_init() {
    #pragma acc parallel loop independent collapse(2) present(U, P, initialHandler)
    for (int i = 0; i < CCX; i ++) {
    for (int j = 0; j < CCY; j ++) {
        U[0][i][j] = initialHandler.mainstream_u;
        U[1][i][j] = initialHandler.mainstream_v;
        P[i][j] = 0;
    }
    }
}

void init() {
    acc_init();
    lsvar.init();
    grid_init();
    inflowHandler.init(0, GC, 2, CY, "data/inflow_boundary");
    initialHandler.init(0, 0, CCX, CCY, "data/initial_field");
    // initialHandler.apply(0, U);
    field_init();
    eq_init();
    interpolation(MAXDIAGI);
    printf("max diag = %lf\n", 1./MAXDIAGI);
}

void finalize() {
    inflowHandler.finalize();
    initialHandler.finalize();
    lsvar.finalize();
    acc_finalize();
}

void output_field(int n) {
    #pragma acc update self(U, P, DIV)
    string fname_base = "data/2dinflow.csv";
    vector<char> fname(fname_base.size() + 32);
    sprintf(fname.data(), "%s.%d", fname_base.c_str(), n);
    FILE *file = fopen(fname.data(), "w");
    fprintf(file, "x,y,z,u,v,w,p,div\n");
    for (int j = GC; j < GC+CY; j ++) {
    for (int i = GC; i < GC+CX; i ++) {
        fprintf(file, "%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e,%12.5e\n", X[i], Y[j], 0., U[0][i][j], U[1][i][j], 0., P[i][j], DIV[i][j]);
    }}
    fclose(file);
}

void main_loop() {
    copy_field(UP[0], U[0]);
    copy_field(UP[1], U[1]);
    prediction();
    periodic_y(U[0], GC);
    periodic_y(U[1], GC);
    interpolation(MAXDIAGI);

    sor_poisson(PMAT, P, RHS, lsvar);
    field_centralize(P);
    pressure_bc();

    projection_center();
    projection_interface();
    velocity_bc();

    calc_divergence();
    max_cfl();
}

int main() {
    init();
    
    for (ISTEP = 0; ISTEP <= MAXSTEP; ISTEP ++) {
        if (ISTEP >= 0) {
            main_loop();
            printf("\r%9d, %10.5lf, %3d, %10.3e, %10.3e, %10.3e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, MAX_CFL);
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

