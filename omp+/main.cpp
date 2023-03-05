// Copyright 2023 Kitaev Pavel

#include "omp.h"
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

void ParallelAlgOMP(double* matrix, int size, double eps) {
    int th_num = 8;
    double* dm = new double[size]{ 0 };
    double dmax;
    const int N = size - 2;
    
    do {
        dmax = 0;
        
#pragma omp parallel for num_threads(th_num)
        for (int wave = 1; wave < N + 1; wave++) {
            dm[wave-1] = 0;
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
        
#pragma omp parallel for num_threads(th_num)
        for (int wave = N - 1; wave > 0; wave--) {
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
        
#pragma omp barrier
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
  double* matrix_omp = new double[size * size];
  FillingTheMatrix(matrix_omp, size);

  double start_omp = omp_get_wtime();
  ParallelAlgOMP(matrix_omp, size, eps);
  double end_omp = omp_get_wtime();

  double time_omp = (end_omp - start_omp).seconds();
  std::cout << "OMP Time: " << time_omp << std::endl;
  //PrintMatrix(matrix_omp, size);
  return 0;
}