#include "test_func.h"

double power(double x, int exp) {
    double r = 1.0;
    while (exp--) {
        r *= x;
    }
    return r;
}