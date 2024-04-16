#include <random>
#include "gaussgen.h"

using namespace std;

default_random_engine GEN;
normal_distribution<double> GAUSS(0., 1.);

double GaussGen::get() {
    return GAUSS(GEN);
}