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

void SequentialAlg(double* matrix, int size, double eps) {
    double* dm = new double[size]{ 0 };
    double dmax, temp, d;
    const int N = size - 2;
    
    do {
        dmax = 0;
        
        for (int wave = 1; wave < N + 1; wave++) {
            dm[wave-1] = 0;
            for (int i = 1; i < wave + 1; i++) {
                int j = wave + 1 - i;
                temp = matrix[size * i + j];
                
                matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                               matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                               matrix[size * (i - 1) + j]);
                
                d = fabs(temp - matrix[size * i + j]);
                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }
        }
        
        for (int wave = N - 1; wave > 0; wave--) {
            for (int i = N - wave + 1; i < N + 1; i++) {
                int j = 2 * N - wave - i + 1;
                temp = matrix[size * i + j];
                
                matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                               matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                               matrix[size * (i - 1) + j]);
                
                d = fabs(temp - matrix[size * i + j]);
                if (dm[i - 1] < d) {
                    dm[i - 1] = d;
                }
            }
        }
        
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
  double* matrix_seq = new double[size * size];
  FillingTheMatrix(matrix_seq, size);

  oneapi::tbb::tick_count start_seq = oneapi::tbb::tick_count::now();
  SequentialAlg(matrix_seq, size, eps);
  oneapi::tbb::tick_count end_seq = oneapi::tbb::tick_count::now();;
  
  double time_seq = (end_seq - start_seq).seconds();
  
  std::cout << "SEQ Time: " << time_seq << std::endl;
  //PrintMatrix(matrix_seq, size);

  return 0;
}