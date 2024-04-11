#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iostream>
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Core"

const static int dim = 2;
const static int nsamples = 10000;

int main() {
    Eigen::VectorXd mean(dim);
    Eigen::MatrixXd covar(dim, dim);

    mean  << 0.0, 0.0;
    covar << 1.0, 0.0, 0.0, 1.0;

    Eigen::LLT<Eigen::MatrixXd> cholesky(covar);
    Eigen::MatrixXd A = cholesky.matrixL();

    std::cout << "Cholesky:\n" << A << std::endl;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> nd(0.0, 1.0);

    FILE *file = fopen("mvgauss.csv", "w");
    fprintf(file, "x,y\n");

    for (int i = 0; i < nsamples; i ++) {
        Eigen::VectorXd seed(dim);
        for (int d = 0; d < dim; d ++) {
            seed(d) = nd(gen);
        }
        Eigen::VectorXd sample(dim);
        sample = mean + A * seed;
        fprintf(file, "%e,%e\n", sample(0), sample(1));
    }
    fclose(file);
    
}
