#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <helper_cuda.h>

extern "C" {
#include <cblas.h>

#define BLOCK_SIZE 16

    void matmult_nat(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                C[n * i + j] = 0.0;
                for (h = 0; h < k; h++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_mnk(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                for (h = 0; h < k; h++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_nmk(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (j = 0; j < n; j++)
        {
            for (i = 0; i < m; i++)
            {
                for (h = 0; h < k; h++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_nkm(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (j = 0; j < n; j++)
        {
            for (h = 0; h < k; h++)
            {
                for (i = 0; i < m; i++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_kmn(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (h = 0; h < k; h++)
        {
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_knm(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (h = 0; h < k; h++)
        {
            for (j = 0; j < n; j++)
            {
                for (i = 0; i < m; i++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_mkn(int m, int n, int k, double *A, double *B, double *C)
    {
        int i, j, h;
        for (i = 0; i < m * n; i++)
        {
            C[i] = 0.0;
        }
        for (i = 0; i < m; i++)
        {
            for (h = 0; h < k; h++)
            {
                for (j = 0; j < n; j++)
                {
                    C[n * i + j] += A[k * i + h] * B[n * h + j];
                }
            }
        }
    }

    void matmult_lib(int m, int n, int k, double *A, double *B, double *C)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    }

    __global__ void matmult_gpu1_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        if (row < m && col < n)
        {
            double sum = 0.0;
            int i;
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
    }

    void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C)
    {
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu1_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu2_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

        double sum = 0.0;
        int i;
        if (row < m && col < n)
        {
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
        row += 1;
        if (row < m && col < n)
        {
	        sum = 0.0;
	        for (i = 0; i < k; i++)
	        {
	        	sum += A[k * row + i] * B[n * i + col];
	        }
	        C[n * row + col] = sum;
	    }
    }

    void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C)
    {
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu2_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu3_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        int col = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
        int row = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

        double sum = 0.0;
        int i;
        if (row < m && col < n)
        {
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
        row += 1;
        if (row < m && col < n)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
        row -= 1;
        col += 1;
        if (row < m && col < n)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
        row += 1;
        if (row < m && col < n)
        {
            sum = 0.0;
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            C[n * row + col] = sum;
        }
    }

    void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C)
    {
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        dim3 DimGrid((n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (m / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu3_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu4_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
	__shared__ double A_sub[BLOCK_SIZE][BLOCK_SIZE];
	//different sizes block indencis
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int row = blockY * BLOCK_SIZE + threadY;
	int col = blockX * BLOCK_SIZE + threadX;
	double sum = 0.0;
	
	int i,j;
	for(i = 0; i < k; i += BLOCK_SIZE) {
		//calculating the A & B blocks
		if (row < m && (i + threadX) < k)
		{
			A_sub[threadY][threadX] = A[(row * k) + i + threadX];
		} else {
			A_sub[threadY][threadX] = 0.0;
		}
		__syncthreads();

		//summing the blocks after they are calculated
		for(j = 0; j < BLOCK_SIZE; ++j) {
			sum += A_sub[threadY][j] * B[(j+i) * n + col];
		}
		__syncthreads();
	}
	if(row < m && col < n) {
		C[row * n + col] = sum;	
	}
    }

    void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C)
    {
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        dim3 DimGrid((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu4_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

	/*
A: m * k
B: k * n
C: m * n
*/
    __global__ void matmult_gpu5_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
	__shared__ double A_sub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double B_sub[BLOCK_SIZE][BLOCK_SIZE];
	//different sizes block indencis
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int row = blockY * BLOCK_SIZE + threadY;
	int col = blockX * BLOCK_SIZE + threadX;
	double sum = 0.0;
	
	int i,j;
	for(i = 0; i < k; i+= BLOCK_SIZE) {
		//calculating the A & B blocks
		if (row < m && (i+threadX) < k)
		{
			A_sub[threadY][threadX] = A[(row * k) + i + threadX];
		} else {
			A_sub[threadY][threadX] = 0.0;
		}	
		if(col < n && (i + threadY) < k) {
			B_sub[threadY][threadX] = B[(i + threadY) * n + col];
		} else {
			B_sub[threadY][threadX] = 0.0;		
		}
		__syncthreads();

		//summing the blocks after they are calculated
		for(j = 0; j < BLOCK_SIZE; ++j) {
			sum += A_sub[threadY][j] * B_sub[j][threadX];		
		}
		__syncthreads();
	}
	if(row < m && col < n) {
		C[row * n + col] = sum;	
	}
    }

    void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C)
    {
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        dim3 DimGrid((n-1)/BLOCK_SIZE+1,(m-1)/BLOCK_SIZE+1);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu5_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C)
    {

        double *device_a, *device_b, *device_c, *trans_C;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));
        trans_C = (double *) malloc(m * n * sizeof(double));

        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0, beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, device_a, k, device_b, n, &beta, device_c, m);
        cublasDestroy(handle);

        checkCudaErrors(cudaMemcpy(trans_C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        int row, col;
        for (row = 0; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                C[n * row + col] = trans_C[m * col + row];
            }
        }

        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }
}
