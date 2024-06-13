#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

struct CommParam {
    static const double RE = 1e4;
    static const double REI = 1./RE;
    static const double SOR_OMEGA = 1.2;
    static const double LS_EPS = 1e-3;
    static const int LS_MAXIRER = 1000;
};

static inline int id(int i, int j, int jmax) {
    return i*jmax + j;
}

struct geomInfo_t {
    int gc = 2;
    int cx, cy;
    int ccx, ccy;
    double lx, ly;
    double dx, dy, dxi, dyi;
    double ddx, ddy, ddxi, ddyi;

    int ccnum() const {
        return ccx*ccy;
    }

    int cnum() const {
        return cx*cy;
    }
};

struct lsInfo_t {
    int iter;
    int error;
};

double diffusion_core(double *phi, int *stencil, int i, int j, const geomInfo_t *geom) {
    const int &ccy = geom->ccy;
    const int ccn = geom->ccnum();
    int idc = id(i,j,ccy);
    int ide = stencil[id(1,idc,ccn)];
    int idw = stencil[id(3,idc,ccn)];
    int idn = stencil[id(5,idc,ccn)];
    int ids = stencil[id(7,idc,ccn)];
    double phic = phi[idc];
    double phie = phi[ide];
    double phiw = phi[idw];
    double phin = phi[idn];
    double phis = phi[ids];
    double dx2 = geom->ddxi*(phie - 2*phic + phiw);
    double dy2 = geom->ddyi*(phin - 2*phic + phis);
    return CommParam::REI*(dx2 + dy2);
}

double kk_advection_core(double *phi, double *u, int *stencil, int i, int j, const geomInfo_t *geom) {
    const int &ccy = geom->ccy;
    const int ccn = geom->ccnum();
    int idcc = id(i,j,ccy);
    int ide1 = stencil[id(1,idcc,ccn)];
    int ide2 = stencil[id(2,idcc,ccn)];
    int idw1 = stencil[id(3,idcc,ccn)];
    int idw2 = stencil[id(4,idcc,ccn)];
    int idn1 = stencil[id(5,idcc,ccn)];
    int idn2 = stencil[id(6,idcc,ccn)];
    int ids1 = stencil[id(7,idcc,ccn)];
    int ids2 = stencil[id(8,idcc,ccn)];
    double phicc = phi[idcc];
    double phie1 = phi[ide1];
    double phie2 = phi[ide2];
    double phiw1 = phi[idw1];
    double phiw2 = phi[idw2];
    double phin1 = phi[idn1];
    double phin2 = phi[idn2];
    double phis1 = phi[ids1];
    double phis2 = phi[ids2];
    double ucc = u[id(0,idcc,ccn)];
    double vcc = u[id(1,idcc,ccn)];
    const double &dxi = geom->dxi;
    const double &dyi = geom->dyi;
    double dx1 = ucc*dxi*(- phie2 + 8*phie1 - 8*phiw1 + phiw2)/12.;
    double dx4 = fabs(ucc)*dxi*.25*(phie2 - 4*phie1 + 6*phicc - 4*phiw1 + phiw2);
    double dy1 = vcc*dyi*(- phin2 + 8*phin1 - 8*phis1 + phis2)/12.;
    double dy4 = fabs(vcc)*dyi*.25*(phin2 - 4*phin1 + 6*phicc - 4*phis1 + phis2);
    return dx1 + dx4 + dy1 + dy4;
}

double riam_adveciton_core(double *phi, double *u, double *uu, int *stencil, int i, int j, const geomInfo_t *geom) {
    const int &ccy = geom->ccy;
    const int ccn = geom->ccnum();
    int idcc = id(i,j,ccy);
    int ide1 = stencil[id(1,idcc,ccn)];
    int ide2 = stencil[id(2,idcc,ccn)];
    int idw1 = stencil[id(3,idcc,ccn)];
    int idw2 = stencil[id(4,idcc,ccn)];
    int idn1 = stencil[id(5,idcc,ccn)];
    int idn2 = stencil[id(6,idcc,ccn)];
    int ids1 = stencil[id(7,idcc,ccn)];
    int ids2 = stencil[id(8,idcc,ccn)];
    double phicc = phi[idcc];
    double phie1 = phi[ide1];
    double phie2 = phi[ide2];
    double phiw1 = phi[idw1];
    double phiw2 = phi[idw2];
    double phin1 = phi[idn1];
    double phin2 = phi[idn2];
    double phis1 = phi[ids1];
    double phis2 = phi[ids2];
    double ucc = u[id(0,idcc,ccn)];
    double vcc = u[id(1,idcc,ccn)];
    double uue = uu[id(0,idcc,ccn)];
    double uuw = uu[id(0,idw1,ccn)];
    double vvn = uu[id(1,idcc,ccn)];
    double vvs = uu[id(1,ids1,ccn)];
    const double &dxi = geom->dxi;
    const double &dyi = geom->dyi;
    double dx1e = (- phie2 + 27*phie1 - 27*phicc + phiw1)*dxi*uue;
    double dx1w = (- phie1 + 27*phicc - 27*phiw1 + phiw2)*dxi*uuw;
    double dy1n = (- phin2 + 27*phin1 - 27*phicc + phis1)*dyi*vvn;
    double dy1s = (- phin1 + 27*phicc - 27*phis1 + phis2)*dyi*vvs;
    double dx4 = (phie2 -4*phie1 + 6*phicc - 4*phiw1 + phiw2)*dxi*fabs(ucc);
    double dy4 = (phin2 -4*phin1 + 6*phicc - 4*phis1 + phis2)*dyi*fabs(vcc);
    return (.5*(dx1e + dx1w + dy1n + dy1s) + (dx4 + dy4))/24;
}

double rbsor_core(double *a, double *x, double *b, int *stencil, int i, int j, const geomInfo_t *geom, int color) {
    if ((i + j)%2 == color) {
        const int &ccy = geom->ccy;
        const int ccn = geom->ccnum();
        int idc = id(i,j,ccy);
        int ide = stencil[id(1,idc,ccn)];
        int idw = stencil[id(3,idc,ccn)];
        int idn = stencil[id(5,idc,ccn)];
        int ids = stencil[id(7,idc,ccn)];
        double ac = a[id(0,idc,ccn)];
        double ae = a[id(1,idc,ccn)];
        double aw = a[id(3,idc,ccn)];
        double an = a[id(5,idc,ccn)];
        double as = a[id(7,idc,ccn)];
        double xc = x[idc];
        double xe = x[ide];
        double xw = x[idw];
        double xn = x[idn];
        double xs = x[ids];
        double cc = (b[idc] - (ac*xc + ae*xe + aw*xw + an*xn + as*xs))/ac;
        x[idc] = xc + CommParam::SOR_OMEGA*cc;
        return cc*cc;
    } else {
        return 0;
    }
}

double sor_poisson(double *a, double *x, double *b, double *r, int *stencil, const geomInfo_t *geom, lsInfo_t *solver) {
    int ccn = geom->ccnum();
    for (solver->iter = 1; solver->iter <= CommParam::LS_MAXIRER; solver->iter ++) {
        #pragma acc parallel loop independent collapse(2) present(a[:ccn*5], x[:ccn], b[:ccn], stencil[:ccn*9], geom[:1])
        for (int i = geom->gc; i < geom->gc + geom->cx; i ++) {
        for (int j = geom->gc; j < geom->gc + geom->cy; j ++) {
            rbsor_core(a, x, b, stencil, i, j, geom, 0);
        }}
        #pragma acc parallel loop independent collapse(2) present(a[:ccn*5], x[:ccn], b[:ccn], stencil[:ccn*9], geom[:1])
        for (int i = geom->gc; i < geom->gc + geom->cx; i ++) {
        for (int j = geom->gc; j < geom->gc + geom->cy; j ++) {
            rbsor_core(a, x, b, stencil, i, j, geom, 1);
        }}
    }
}






