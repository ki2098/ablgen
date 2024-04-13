#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <random>
#include "fftw3.h"

using namespace std;

const static int NSAMPLES = 25;

const static int REAL = 0;
const static int IMAG = 1;
const static double Pi = M_PI;

#define SQ(x) ((x)*(x))

void generate_signal(fftw_complex *signal) {
    for (int i = 0; i < NSAMPLES; i ++) {
        double theta = (i/double(NSAMPLES))*2.*M_PI;
        signal[i][REAL] = 1.*cos(2.*theta) + .5*cos(4.*theta) + .25*cos(8.*theta);
        signal[i][IMAG] = 1.*sin(2.*theta) + .5*sin(4.*theta) + .25*sin(8.*theta);
    }
}

void scale_complex_seq(fftw_complex *seq, int size, double scale) {
    for (int i = 0; i < size; i ++) {
        seq[i][REAL] *= scale;
        seq[i][IMAG] *= scale;
    }
}

enum class SIGN {
    FORWARD = -1,
    BACKWARD = 1
};

void dft(fftw_complex *in, fftw_complex *out, int size, SIGN sign) {
    for (int m = 0; m < size; m ++) {
        out[m][REAL] = out[m][IMAG] = 0.;
        for (int n = 0; n < size; n ++) {
            double theta = double(sign)*2.*Pi*m*n/double(size);
            out[m][REAL] += in[n][REAL]*cos(theta) - in[n][IMAG]*sin(theta);
            out[m][IMAG] += in[n][REAL]*sin(theta) + in[n][IMAG]*cos(theta);
        }
    }
}

void print_all(fftw_complex *signal, fftw_complex *rbuild, fftw_complex *series, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%5d| %8.3lf + i %8.3lf| %8.3lf + i %8.3lf| %8.3lf + i %8.3lf\n", i, signal[i][REAL], signal[i][IMAG], rbuild[i][REAL], rbuild[i][IMAG], series[i][REAL], series[i][IMAG]);
    }
}

int main() {
    fftw_complex signal[NSAMPLES];
    fftw_complex series[NSAMPLES];
    fftw_complex rbuild[NSAMPLES];

    generate_signal(signal);
    dft(signal, series, NSAMPLES, SIGN::FORWARD);
    scale_complex_seq(series, NSAMPLES, 1./NSAMPLES);
    dft(series, rbuild, NSAMPLES, SIGN::BACKWARD);

    print_all(signal, rbuild, series, NSAMPLES);
}