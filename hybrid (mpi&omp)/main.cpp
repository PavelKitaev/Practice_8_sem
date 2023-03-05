// Copyright 2023 Kitaev Pavel

#include "mpi.h"
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

void ParallelAlgHybrid(double* matrix, int size, double eps)
{
    int q = 0;
    
    int procRank, procNum;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    
    double* dm;
    MPI_Win win_dm;
    if (procRank == 0) {
        MPI_Win_allocate_shared(size * sizeof(double), 0, MPI_INFO_NULL, MPI_COMM_WORLD, &dm, &win_dm);
    } else {
        int disp;
        MPI_Aint ssize;
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &dm, &win_dm);
        MPI_Win_shared_query(win_dm, 0, &ssize, &disp, &dm);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double dmax;
    
    const int delta = size / procNum;
    int residue = size % procNum;
    
    int* i_start_inc = new int[procNum];
    int* i_end_inc = new int[procNum];
    int* i_start_dec = new int[procNum];
    int* i_end_dec = new int[procNum];
    
    if (procRank == 0)
    {
        i_start_inc[0] = 1;
        i_end_inc[0] = 0;
        
        i_start_dec[0] = (size - 2) - 1;
        i_end_dec[0] = (size - 2) - 1;
        
        for (int k = 0; k < procNum; k++) {
            i_end_inc[k] += (delta);
            i_end_dec[k] -= (delta);
            
            if (residue > 0) {
                i_end_dec[k]--;
                
                i_end_inc[k]++;
                residue--;
            }
            
            if (k == procNum - 1) {
                i_end_inc[k] = size - 1;
                if (procNum == 2) {
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
    }
    
    MPI_Bcast(&i_start_inc[0], procNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_end_inc[0], procNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_start_dec[0], procNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&i_end_dec[0], procNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    do
    {   
#pragma omp parallel for num_threads(2)
        for (int wave = i_start_inc[procRank]; wave < i_end_inc[procRank]; wave++) {
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
#pragma omp parallel for num_threads(2)
        for (int wave = i_start_dec[procRank]; wave > i_end_dec[procRank]; wave--) {
            for (int i = (size - 2) - wave + 1; i < (size - 2) + 1; i++) {
                int j = 2 * (size - 2) - wave - i + 1;
                
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
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (procRank == 0) {
            dmax = 0;
            for (int i = 0; i < size; i++) {
                if (dmax < dm[i]) {
                    dmax = dm[i];
                }
            }
        }
        
        MPI_Bcast(&dmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } while (dmax > eps);
    
    delete[] i_start_inc;
    delete[] i_end_inc;
    delete[] i_start_dec;
    delete[] i_end_dec;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win_dm);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int procRank, procNum;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    
    int size = 3000;
    double eps = 0.01;
    
    double start_mpi, end_mpi;
    double* matrix_mpi;
    MPI_Win win_matrix;      //Создаем окно доступа к матрице

    if (procRank == 0) {
        MPI_Win_allocate_shared(size*size * sizeof(double), 0, MPI_INFO_NULL, MPI_COMM_WORLD, &matrix_mpi, &win_matrix);
        FillingTheMatrix(matrix_mpi, size);
        
        start_mpi = MPI_Wtime();
    }
    else {
        int disp;
        MPI_Aint ssize;
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &matrix_mpi, &win_matrix);
        MPI_Win_shared_query(win_matrix, 0, &ssize, &disp, &matrix_mpi);
    }
    
    ParallelAlgHybrid(matrix_mpi, size, eps);
    
    if (procRank == 0){
        end_mpi = MPI_Wtime();
        double time_mpi = end_mpi - start_mpi;
        
        std::cout << "MPI Hybrid time: " << time_mpi << std::endl;
        //PrintMatrix(matrix_mpi, size);   
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_free(&win_matrix);
    
    MPI_Finalize();

  return 0;
}