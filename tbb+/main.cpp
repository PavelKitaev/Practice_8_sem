// Copyright 2023 Kitaev Pavel

#include "oneapi/tbb.h"
#include <iostream>
#include <cmath>

void PrintMatrix(double* matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << matrix[i * size + j] << " ";
    }

    std::cout << std::endl;
  }
}

void FillingTheMatrix(double* matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if ((i == 0) || (i == size - 1) || (j == 0) || (j == size - 1)) {
        matrix[i * size + j] = 100;
      } else {
        matrix[i * size + j] = 0;
      }
    }
  }
}

void ParallelAlgTBB(double* matrix, int size, double eps) {
    oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 8);
    
    double* dm = new double[size] { 0 };
    double dmax;
    const int N = size - 2;
    
    do {
        dmax = 0;
        tbb::parallel_for(tbb::blocked_range<int>(1, N + 1, 1),
                          [&](const tbb::blocked_range<int>& range) {
            for (int wave = range.begin(); wave < range.end(); wave++) {
                dm[wave - 1] = 0;
                for (int i = 1; i < wave + 1; i++) {
                    int j = wave + 1 - i;
                    double temp = matrix[size * i + j];
                    
                    matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                                   matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                                   matrix[size * (i - 1) + j]);
                    
                    double d = fabs(temp - matrix[size * i + j]);
                    
                    if (dm[i - 1] < d) {
                        dm[i - 1] = d;
                    }
                }
            }
        });
        
        tbb::parallel_for(tbb::blocked_range<int>(0, N - 1, 1),
                          [&](const tbb::blocked_range<int>& range) {
            for (int wave = range.end(); wave > range.begin(); wave--) {
                for (int i = N - wave + 1; i < N + 1; i++) {
                    int j = 2 * N - wave - i + 1;
                    double temp = matrix[size * i + j];
                    
                    matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                                   matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                                   matrix[size * (i - 1) + j]);
                    
                    double d = fabs(temp - matrix[size * i + j]);
                    
                    if (dm[i - 1] < d) {
                        dm[i - 1] = d;
                    }
                }
            }
        });
        
        for (int i = 0; i < size; i++) {
            if (dmax < dm[i]) {
                dmax = dm[i];
            }
        }
    } while (dmax > eps);
    
    delete[] dm;
}

int main(int argc, char **argv)
{
  double* matrix_tbb = new double[size * size];
  FillingTheMatrix(matrix_tbb, size);
  oneapi::tbb::tick_count start_tbb = oneapi::tbb::tick_count::now();
  ParallelAlgTBB(matrix_tbb, size, eps);
  oneapi::tbb::tick_count end_tbb = oneapi::tbb::tick_count::now();
  
  double time_tbb = (end_tbb - start_tbb).seconds();
  std::cout << "TBB Time: " << time_tbb << std::endl;
  //PrintMatrix(matrix_tbb, size);

  return 0;
}