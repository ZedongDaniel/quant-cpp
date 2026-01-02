#include "test_func.h"
#include <iostream>
#include <cmath>
#include <omp.h>

double power(double x, int exp) {
    return std::pow(x, exp);
}

void test_openmp(int num_threads) {
    std::cout << "========================================" << std::endl;
    std::cout << "OpenMP Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int max_threads = omp_get_max_threads();
    std::cout << "Max threads available: " << max_threads << std::endl;
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        std::cout << "Requested threads: " << num_threads << std::endl;
    } else {
        std::cout << "Using default thread count" << std::endl;
    }
    
    int actual_threads;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    std::cout << "Actually using: " << actual_threads << " threads" << std::endl;
    
    std::cout << "\nTesting parallel execution:" << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        #pragma omp critical
        {
            std::cout << "  Thread " << omp_get_thread_num() 
                      << " processing iteration " << i << std::endl;
        }
    }
    std::cout << "========================================" << std::endl;
}

int get_num_threads() {
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}