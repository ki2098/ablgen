#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <random>

using namespace std;

const static int CX  = 50;
const static int CY  = 50;
const static int GC  = 3;
const static int CCX = CX + 2*GC;
const static int CCY = CY + 2*GC;
const static double LX = 1.0;
const static double LY = 1.0;
const static double DX = LX/CX;
const static double DY = LY/CY;
const static double DDX = DX*DX;
const static double DDY = DY*DY;
const static double DXI = 1./DX;
const static double DYI = 1./DY;
const static double DDXI = 1./DDX;
const static double DDYI = 1./DDY;
const static double DT = 1e-3;
const static double DTI = 1./DT;

const static double RE = 1e4;
const static double REI = 1./RE;
const static double SOR_OMEGA = 1.0;
int LS_ITER;
const static int LS_MAXITER = 1000;
const static double LS_EPS = 1e-6;
double LS_ERR;
int ISTEP;
int MAXSTEP = int(100./DT);
double RMS_DIV;

const static double UTSAVG = 0.0;
const static double VTSAVG = 0.0;
const static double FORCING_EPS = 1e-3;
double TURB_INTEN;
double TURB_K;

double X[CCX]={}, Y[CCX]={};
double U[2][CCX][CCY]={}, UU[2][CCX][CCY]={}, P[CCX][CCY]={}, UP[2][CCX][CCY]={};
double RHS[CCX][CCY]={};
double FF[2][CCX][CCY];
double NUT[CCX][CCY];
const static double C_SMAGORINSKY = 0.2;

inline double gettime() {
    return ISTEP*DT;
}

static inline double sq(double a) {
    return a*a;
}

double advection_core(double phi[CCX][CCY], double uu[CCX][CCY], double vv[CCX][CCY], double u[CCX][CCY], double v[CCX][CCY], int i, int j) {
    double phicc = phi[i  ][j  ];
    double phie1 = phi[i+1][j  ];
    double phie2 = phi[i+2][j  ];
    double phiw1 = phi[i-1][j  ];
    double phiw2 = phi[i-2][j  ];
    double phin1 = phi[i  ][j+1];
    double phin2 = phi[i  ][j+2];
    double phis1 = phi[i  ][j-1];
    double phis2 = phi[i  ][j-2];
    double uE    =  uu[i  ][j  ];
    double uW    =  uu[i-1][j  ];
    double vN    =  vv[i  ][j  ];
    double vS    =  vv[i  ][j-1];
    double ucc   =   u[i  ][j  ];
    double vcc   =   v[i  ][j  ];
    double phixE = - phie2 + 27*(phie1 - phicc) + phiw1;
    double phixW = - phie1 + 27*(phicc - phiw1) + phiw2;
    double phiyN = - phin2 + 27*(phin1 - phicc) + phis1;
    double phiyS = - phin1 + 27*(phicc - phis1) + phis2;
    double phi4x = phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2;
    double phi4y = phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2;
    double axE   = uE*phixE*DXI;
    double axW   = uW*phixW*DXI;
    double ayN   = vN*phiyN*DYI;
    double ayS   = vS*phiyS*DYI;
    double adv   = (0.5*(axE + axW + ayN + ayS) + (fabs(ucc)*phi4x + fabs(vcc)*phi4y))/24.;
    return adv;
}

double diffusion_core(double phi[CCX][CCY], double nut[CCX][CCY], int i, int j) {
    double dphiE = DXI*(phi[i+1][j  ] - phi[i  ][j  ]);
    double dphiW = DXI*(phi[i  ][j  ] - phi[i-1][j  ]);
    double dphiN = DYI*(phi[i  ][j+1] - phi[i  ][j  ]);
    double dphiS = DYI*(phi[i  ][j  ] - phi[i  ][j-1]);
    double nutE  =  .5*(nut[i+1][j  ] + nut[i  ][j  ]);
    double nutW  =  .5*(nut[i  ][j  ] + nut[i-1][j  ]);
    double nutN  =  .5*(nut[i  ][j+1] + nut[i  ][j  ]);
    double nutS  =  .5*(nut[i  ][j  ] + nut[i  ][j-1]);
    double difx  = DXI*((REI + nutE)*dphiE - (REI + nutW)*dphiW);
    double dify  = DYI*((REI + nutN)*dphiN - (REI + nutS)*dphiS);
    return difx + dify;
}

void prediction() {
    memcpy(&UP[0][0][0], &U[0][0][0], sizeof(double)*2*CCX*CCY);
    for (int m = 0; m < 2; m ++) {
        #pragma omp parallel for collapse(2)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            double adv = advection_core(UP[m], UU[0], UU[1], UP[0], UP[1], i, j);
            double dif = diffusion_core(UP[m], NUT, i, j);
            U[m][i][j] = UP[m][i][j] + DT*(- adv + dif + FF[m][i][j]);
        }}
    }   
}

void interpolation() {
    #pragma omp parallel for collapse(2)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] = .5*(U[0][i][j] + U[0][i+1][j]);
    }}
    #pragma omp parallel for collapse(2)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[1][i][j] = .5*(U[1][i][j] + U[1][i][j+1]);
    }}
    #pragma omp parallel for collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        RHS[i][j] = DTI*(DXI*(UU[0][i][j] - UU[0][i-1][j]) + DYI*(UU[1][i][j] - UU[1][i][j-1]));
    }}
}

double sor_serial_core(double phi[CCX][CCY], double rhs[CCX][CCY], int i, int j) {
    double phiage = (phi[i+1][j] + phi[i-1][j])*DDXI + (phi[i][j+1] + phi[i][j-1])*DDYI + rhs[i][j];
    double dphi = 0.5*phiage/(DDXI + DDYI) - phi[i][j];
    phi[i][j] += SOR_OMEGA*dphi;
    return sq(dphi);
}

double sor_rb_core(double phi[CCX][CCY], double rhs[CCX][CCY], int i, int j, int color) {
    if ((i + j)%2 == color) {
        double cc = (rhs[i][j] - (DDXI*(phi[i+1][j] - 2*phi[i][j] + phi[i-1][j]) + DDYI*(phi[i][j+1] - 2*phi[i][j] + phi[i][j-1])))/(- 2*DDXI - 2*DDYI);
        phi[i][j] += SOR_OMEGA*cc;
        return sq(cc);
    } else {
        return 0;
    }
}

void pressure_bc() {
    for (int j = GC; j < GC+CY; j ++) {
        P[GC -1][j] = P[GC+CX-1][j];
        P[GC+CX][j] = P[GC     ][j];
    }
    for (int i = GC; i < GC+CX; i ++) {
        P[i][GC-1 ] = P[i][GC+CX-1];
        P[i][GC+CY] = P[i][GC     ];
    }
}

void ls_poisson() {
    for(LS_ITER = 1; LS_ITER <= LS_MAXITER; LS_ITER ++) {
        LS_ERR = 0;
        #pragma omp parallel for reduction(+:LS_ERR) collapse(2)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            LS_ERR += sor_rb_core(P, RHS, i, j, 0);
        }}
        pressure_bc();
        #pragma omp parallel for reduction(+:LS_ERR) collapse(2)
        for (int i = GC; i < GC+CX; i ++) {
        for (int j = GC; j < GC+CY; j ++) {
            LS_ERR += sor_rb_core(P, RHS, i, j, 1);
        }}
        pressure_bc();
        LS_ERR = sqrt(LS_ERR/(CX*CY));
        if (LS_ERR < LS_EPS) {
            return;
        }
    }
}

void pressure_centralize() {
    double sum = 0;
    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        sum += P[i][j];
    }}
    
    double avg = sum / double(CX*CY);
    #pragma omp parallel for collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        P[i][j] -= avg;
    }}
}

void projection_center() {
    #pragma omp parallel for collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        U[0][i][j] -= .5*DT*DXI*(P[i+1][j] - P[i-1][j]);
        U[1][i][j] -= .5*DT*DYI*(P[i][j+1] - P[i][j-1]);
    }}
}

void projection_interface() {
    #pragma omp parallel for collapse(2)
    for (int i = GC-1; i < GC+CX; i ++) {
    for (int j = GC  ; j < GC+CY; j ++) {
        UU[0][i][j] -= DT*DX*(P[i+1][j] - P[i][j]);
    }}
    #pragma omp parallel for collapse(2)
    for (int i = GC  ; i < GC+CX; i ++) {
    for (int j = GC-1; j < GC+CY; j ++) {
        UU[1][i][j] -= DT*DY*(P[i][j+1] - P[i][j]);
    }}
    RMS_DIV = 0;
    #pragma omp parallel for reduction(+:RMS_DIV) collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        RMS_DIV += sq(DXI*(UU[0][i][j] - UU[0][i-1][j]) + DYI*(UU[1][i][j] - UU[1][i][j-1]));
    }}
    RMS_DIV = sqrt(RMS_DIV/(CX*CY));
}

void turbulence_core(double u[CCX][CCY], double v[CCX][CCY], double nut[CCX][CCY], int i, int j) {
    double ue   = u[i+1][j];
    double uw   = u[i-1][j];
    double un   = u[i][j+1];
    double us   = u[i][j-1];
    double ve   = v[i+1][j];
    double vw   = v[i-1][j];
    double vn   = v[i][j+1];
    double vs   = v[i][j-1];
    double dudx = .5*DXI*(ue - uw);
    double dudy = .5*DXI*(un - us);
    double dvdx = .5*DYI*(ve - vw);
    double dvdy = .5*DYI*(vn - vs);
    double Du   = sqrt(2*sq(dudx) + 2*sq(dvdy) + sq(dudy + dvdx));
    double De   = sqrt(DX*DY);
    double Lc   = C_SMAGORINSKY*De;
    nut[i][j]   = sq(Lc)*Du;
}

void turbulence() {
    #pragma omp parallel for collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        turbulence_core(U[0], U[1], NUT, i, j);
    }}
}

void forcing_core(double forcex[CCX][CCY], double forcey[CCX][CCY], double u[CCX][CCY], double v[CCX][CCY], double uavg, double vavg, int i, int j) {
    double aiso = FORCING_EPS/(sq(u[i][j]) + sq(v[i][j]));
    forcex[i][j] = aiso*(u[i][j] - uavg);
    forcey[i][j] = aiso*(v[i][j] - vavg);
}

void forcing() {
    #pragma omp parallel for collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        forcing_core(FF[0], FF[1], U[0], U[1], UTSAVG, VTSAVG, i, j);
    }}
}

void velocity_bc() {
    for (int m = 0; m < 2; m ++) {
        for (int j = GC; j < GC+CY; j ++) {
            U[m][GC   -1][j] = U[m][GC+CX-1][j];
            U[m][GC   -2][j] = U[m][GC+CX-2][j];
            U[m][GC+CX  ][j] = U[m][GC     ][j];
            U[m][GC+CX+1][j] = U[m][GC   +1][j];
        }
        for (int i = GC; i < GC+CX; i ++) {
            U[m][i][GC   -1] = U[m][i][GC+CY-1];
            U[m][i][GC   -2] = U[m][i][GC+CY-2];
            U[m][i][GC+CY  ] = U[m][i][GC     ];
            U[m][i][GC+CY+1] = U[m][i][GC   +1];
        }
    }
}

void nut_bc() {
    for (int j = GC; j < GC+CY; j ++) {
        NUT[GC -1][j] = NUT[GC+CX-1][j];
        NUT[GC+CX][j] = NUT[GC     ][j];
    }
    for (int i = GC; i < GC+CX; i ++) {
        NUT[i][GC-1 ] = NUT[i][GC+CX-1];
        NUT[i][GC+CY] = NUT[i][GC     ];
    }
}

void turbulence_intensity() {
    double tisum = 0;
    #pragma omp parallel for reduction(+:tisum) collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double du = U[0][i][j] - UTSAVG;
        double dv = U[1][i][j] - VTSAVG;
        tisum += sqrt(sq(du) + sq(dv))/sqrt(sq(UTSAVG) + sq(VTSAVG));
    }}
    TURB_INTEN = tisum / (CX * CY);
}

void turbulence_kinetic_energy() {
    double ksum = 0;
    #pragma omp parallel for reduction(+:ksum) collapse(2)
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double du = U[0][i][j] - UTSAVG;
        double dv = U[1][i][j] - VTSAVG;
        ksum += .5*(sq(du) + sq(dv));
    }}
    TURB_K = ksum / (CX*CY);
}

void main_loop() {
    prediction();
    velocity_bc();
    interpolation();

    ls_poisson();
    pressure_centralize();
    // pressure_bc();

    projection_center();
    projection_interface();
    velocity_bc();

    turbulence();
    nut_bc();

    forcing();
    turbulence_kinetic_energy();
}

void build_grid() {
    for (int i = 0; i < CCX; i ++) {
        X[i] = (i - GC + .5)*DX;
    }
    for (int j = 0; j < CCY; j ++) {
        Y[j] = (j - GC + .5)*DY;
    }
}

void init_velocity() {
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 0.01);
    for (int i = GC; i < GC+CX; i ++) {
    for (int j = GC; j < GC+CY; j ++) {
        double upert = distribution(generator);
        double vpert = distribution(generator);
        U[0][i][j] = UTSAVG + upert;
        U[1][i][j] = VTSAVG + vpert;
    }}
    velocity_bc();
    interpolation();

    ls_poisson();
    pressure_centralize();
    // pressure_bc();

    projection_center();
    projection_interface();
    velocity_bc();

    turbulence();
    nut_bc();

    forcing();
}

void output(int count) {
    char filename[128];
    sprintf(filename, "data/forcing-test.csv.%d", count);
    FILE *file = fopen(filename, "w");
    fprintf(file, "x,y,z,u,v,w,p\n");
    for (int j = GC; j < GC+CX; j ++) {
    for (int i = GC; i < GC+CY; i ++) {
        fprintf(file, "%10e,%10e,%10e,%10e,%10e,%10e,%10e\n", X[i], Y[j], 0.0, U[0][i][j], U[1][i][j], 0.0, P[i][j]);
    }}
    fclose(file);
}

int main() {
    build_grid();
    init_velocity();
    output(0);
    for (ISTEP = 1; ISTEP <= MAXSTEP; ISTEP ++) {
        main_loop();
        printf("\r%9d, %15lf, %3d, %15e, %15e, %15e", ISTEP, gettime(), LS_ITER, LS_ERR, RMS_DIV, TURB_K);
        fflush(stdout);
        if (ISTEP % int(1./DT) == 0) {
            output(ISTEP / int(1./DT));
        }
    }
    printf("\n");
    return 0;
}
