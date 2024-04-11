#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fftw3.h"

const static int N_SAMPLES = 25;

const static int REAL = 0;
const static int IMAG = 1;

using namespace std;

#define SQ(x) ((x)*(x))

void generate_signal(fftw_complex *signal) {
    for (int i = 0; i < N_SAMPLES; i ++) {
        double theta = (i/double(N_SAMPLES))*2.*M_PI;
        signal[i][REAL] = 1.*cos(2.*theta) + .5*cos(4.*theta) + .25*cos(8.*theta);
        signal[i][IMAG] = 1.*sin(2.*theta) + .5*sin(4.*theta) + .25*sin(8.*theta);
    }
}

void print_complex_seq(fftw_complex *seq, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%5d %12.3e + i %12.3e\n", i, seq[i][REAL], seq[i][IMAG]);
    }
}

void print_all(fftw_complex *signal, fftw_complex *reconstruct, fftw_complex *series, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%5d| %8.3lf + i %8.3lf| %8.3lf + i %8.3lf| %8.3lf + i %8.3lf\n", i, signal[i][REAL], signal[i][IMAG], reconstruct[i][REAL], reconstruct[i][IMAG], series[i][REAL], series[i][IMAG]);
    }
}

void scale_complex_seq(fftw_complex *seq, int size, double scale) {
    for (int i = 0; i < size; i ++) {
        seq[i][REAL] *= scale;
        seq[i][IMAG] *= scale;
    }
}


int main() {
    fftw_complex signal[N_SAMPLES];
    fftw_complex result[N_SAMPLES];
    fftw_complex recons[N_SAMPLES];
    fftw_plan plan = fftw_plan_dft_1d(N_SAMPLES, signal, result, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan invr = fftw_plan_dft_1d(N_SAMPLES, result, recons, FFTW_BACKWARD, FFTW_ESTIMATE);

    generate_signal(signal);
    fftw_execute(plan);
    scale_complex_seq(result, N_SAMPLES, 1./N_SAMPLES);
    fftw_execute(invr);
    print_all(signal, recons, result, N_SAMPLES);

    fftw_destroy_plan(plan);

    return 0;
}