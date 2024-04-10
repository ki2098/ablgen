#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fftw3.h"

const static int N_SAMPLES = 100;

const static int REAL = 0;
const static int IMAG = 1;

using namespace std;

#define SQ(x) ((x)*(x))

void generate_signal(fftw_complex *signal) {
    for (int i = 0; i < N_SAMPLES; i ++) {
        double theta = i / double(N_SAMPLES*M_PI);
        signal[i][REAL] = 1.*cos(10.*theta) + .5*cos(25.*theta);
        signal[i][IMAG] = 1.*sin(10.*theta) + .5*sin(25.*theta);
    }
}

void print_complex_seq(fftw_complex *seq, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%5d %12.3e + i %12.3e\n", i, seq[i][REAL], seq[i][IMAG]);
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
    printf("original series:\n");
    print_complex_seq(signal, N_SAMPLES);
    printf("reconstructed series:\n");
    print_complex_seq(recons, N_SAMPLES);

    fftw_destroy_plan(plan);

    return 0;
}