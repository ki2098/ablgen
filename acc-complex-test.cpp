#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>

#define PI M_PI

const static int N    = 50;
const static int REAL = 0;
const static int IMAG = 1;

typedef double complex[2];

complex carr[N];

void init_complex_array(complex *complex_arr) {
    #pragma acc kernels loop independent present(complex_arr[:N])
    for (int i = 0; i < N; i ++) {
        double th = i*2.*PI/N;
        complex_arr[i][REAL] = cos(th);
        complex_arr[i][IMAG] = sin(th);
    }
}

int main() {
    #pragma acc enter data copyin(carr)
    init_complex_array(carr);
    #pragma acc update self(carr)
    for (int i = 0; i < N; i ++) {
        printf("%6.3lf %6.3lf\n", carr[i][REAL], carr[i][IMAG]);
    }
    cudaMemcpy(&carr[0], &carr[1], sizeof(complex), cudaMemcpyHostToHost);
    printf("\n");
    for (int i = 0; i < N; i ++) {
        printf("%6.3lf %6.3lf\n", carr[i][REAL], carr[i][IMAG]);
    }
    #pragma acc exit data delete(carr)
}