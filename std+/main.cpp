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

void ParallelAlgSTD(double* matrix, int size, double eps) {
    const int th_num = 8; //std::thread::hardware_concurrency();
    std::thread* threads = new std::thread[th_num];
    
    double* dm = new double[size] { 0 };
    double dmax;
    
    const int delta = size / th_num;
    int residue = size % th_num;
    
    int* i_start_inc = new int[th_num];
    int* i_end_inc = new int[th_num];
    int* i_start_dec = new int[th_num];
    int* i_end_dec = new int[th_num];
    
    
    i_start_inc[0] = 1;
    i_end_inc[0] = 0;
    
    i_start_dec[0] = (size - 2) - 1;
    i_end_dec[0] = (size - 2) - 1;
    
    
    for (int k = 0; k < th_num; k++) {
        i_end_inc[k] += (delta);
        i_end_dec[k] -= (delta);
        
        if (residue > 0) {
            i_end_dec[k]--;
            i_end_inc[k]++;
            
            residue--;
        }
        
        if (k == th_num - 1) {
            i_end_inc[k] = size - 1;
            if (th_num == 2) {
                i_start_dec[k]++;
                i_end_dec[k] = i_end_dec[k-1] - delta + 1;
                i_end_dec[k-1]++;
            } else {
                i_end_dec[k] = size-1;
            }
        } else {
            i_start_inc[k+1] = i_end_inc[k];
            i_start_dec[k+1] = i_end_dec[k];
            
            i_end_inc[k+1] = i_end_inc[k];
            i_end_dec[k+1] = i_end_dec[k];
        }
    }
    
    do {
        dmax = 0;
        
        for (int k = 0; k < th_num; k++) {
            threads[k] = std::thread([&](int i_start_inc, int i_end_inc,
                                         int i_start_dec, int i_end_dec) {
                
                for (int wave = i_start_inc; wave < i_end_inc; wave++) {
                    dm[wave - 1] = 0;
                    for (int i = 1; i < wave + 1; i++) {
                        int j = wave + 1 - i;
                        double temp = matrix[size * i + j];
                        
                        matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                                       matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                                       matrix[size * (i - 1) + j]);
                        
                        if (dm[i - 1] < fabs(temp - matrix[size * i + j])) {
                            dm[i - 1] = fabs(temp - matrix[size * i + j]);
                        }
                    }
                }
                
                for (int wave = i_start_dec; wave > i_end_dec; wave--) {
                    for (int i = (size - 2) - wave + 1; i < (size - 2) + 1; i++) {
                        int j = 2 * (size - 2) - wave - i + 1;
                        
                        double temp = matrix[size * i + j];
                        
                        matrix[size * i + j] = 0.25 * (matrix[size * i + j + 1] +
                                                       matrix[size * i + j - 1] + matrix[size * (i + 1) + j] +
                                                       matrix[size * (i - 1) + j]);
                        
                        if (dm[i - 1] < fabs(temp - matrix[size * i + j])) {
                            dm[i - 1] = fabs(temp - matrix[size * i + j]);
                        }
                    }
                }
            }, i_start_inc[k], i_end_inc[k], i_start_dec[k], i_end_dec[k]);
        }
        
        for (int i = 0; i < th_num; i++) {
            threads[i].join();
        }
        
        for (int i = 0; i < size; i++) {
            if (dmax < dm[i]) {
                dmax = dm[i];
            }
        }
        
    } while (dmax > eps);
    
    delete[] threads;
    delete[] dm;
    
    delete[] i_start_inc;
    delete[] i_end_inc;
    delete[] i_start_dec;
    delete[] i_end_dec;
}

int main(int argc, char **argv)
{
  double* matrix_std = new double[size * size];
  FillingTheMatrix(matrix_std, size);
  oneapi::tbb::tick_count start_std = oneapi::tbb::tick_count::now();
  ParallelAlgSTD(matrix_std, size, eps);
  oneapi::tbb::tick_count end_std = oneapi::tbb::tick_count::now();
  double time_std = (end_std - start_std).seconds();
  std::cout << "STD Time: " << time_std << std::endl;
  //PrintMatrix(matrix_std, size);

  return 0;
}