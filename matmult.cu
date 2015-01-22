#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <helper_cuda.h>

extern "C" {
#include <cblas.h>

#define BLOCK_SIZE 16

    void matmult_lib(int m, int n, int k, double *A, double *B, double *C)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    }

    __global__ void matmult_gpu1_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        // Define current row and column
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        // Keeping inside the boundaries
        if (row < m && col < n)
        {
            double sum = 0.0;
            int i;
            // Sum all the values in A's row and B's column
            for (i = 0; i < k; i++)
            {
                sum += A[k * row + i] * B[n * i + col];
            }
            // Add the total value to C
            C[n * row + col] = sum;
        }
    }

    void matmult_gpu1(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Set grid and block size
        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Run kernel
        matmult_gpu1_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        // Synchronize threads
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu2_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        // Define current row and column
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = (threadIdx.y + blockIdx.y * blockDim.y) * 2;

        double sum = 0.0, sum2 = 0.0, tmp;
        int i;
        // Keeping inside the boundaries
        if (row < m && col < n)
        {
            // Sum all the values in A's row and B's column
            for (i = 0; i < k; i++)
            {
                tmp = B[n * i + col];
                sum += A[k * row + i] * tmp;
                sum2 += A[k * (row + 1) + i] * tmp;
            }
            // Add the total value to C
            C[n * row + col] = sum;
            C[n * (row + 1) + col] = sum2;
        }
    }

    void matmult_gpu2(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Set grid and block size
        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Run kernel
        matmult_gpu2_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        // Synchronize threads
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu3_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        // Define current row and column
        int col = (threadIdx.x + blockIdx.x * blockDim.x);
        int row = (threadIdx.y + blockIdx.y * blockDim.y) * 4;

        double sum = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, tmp;
        int i;
        // Keeping inside the boundaries
        if (row < m && col < n)
        {
            // Sum all the values in A's row and B's column
            for (i = 0; i < k; i++)
            {
                tmp = B[n * i + col];
                sum += A[k * row + i] * tmp;
                sum2 += A[k * (row + 1) + i] * tmp;
                sum3 += A[k * (row + 2) + i] * tmp;
                sum4 += A[k * (row + 3) + i] * tmp;
            }
            // Add the total value to C
            C[n * row + col] = sum;
            C[n * (row + 1) + col] = sum2;
            C[n * (row + 2) + col] = sum3;
            C[n * (row + 3) + col] = sum4;
        }
    }

    void matmult_gpu3(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Set grid and block size
        dim3 DimGrid((n / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Run kernel
        matmult_gpu3_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        // Synchronize threads
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu4_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        // Define shared memory for A block
        __shared__ double A_s[BLOCK_SIZE][BLOCK_SIZE];

        // Define current row and column
        int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        double sum = 0.0;

        int i, j;
        for (i = 0; i < k; i += BLOCK_SIZE)
        {
            // Calculate values in A block in shared memory
            if (row < m && (i + threadIdx.x) < k)
            {
                A_s[threadIdx.y][threadIdx.x] = A[(k * row) + i + threadIdx.x];
            }
            else
            {
                A_s[threadIdx.y][threadIdx.x] = 0.0;
            }
            // Synchronize threads
            __syncthreads();

            // Sum up values from A block and B matrix
            for (j = 0; j < BLOCK_SIZE; ++j)
            {
                sum += A_s[threadIdx.y][j] * B[(j + i) * n + col];
            }
            // Synchronize threads
            __syncthreads();
        }
        // Keeping inside the boundaries
        if (row < m && col < n)
        {
            // Add the total value to C
            C[row * n + col] = sum;
        }
    }

    void matmult_gpu4(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Set grid and block size
        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Run kernel
        matmult_gpu4_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        // Synchronize threads
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    __global__ void matmult_gpu5_kernel(int m, int n, int k, double *A, double *B, double *C)
    {
        // Define shared memory for A and B blocks
        __shared__ double A_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double B_s[BLOCK_SIZE][BLOCK_SIZE];

        // Define current row and column
        int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        double sum = 0.0;

        int i, j;
        for (i = 0; i < k; i += BLOCK_SIZE)
        {
            // Calculate values in A and B blocks in shared memory
            if (row < m && (i + threadIdx.x) < k)
            {
                A_s[threadIdx.y][threadIdx.x] = A[(k * row) + i + threadIdx.x];
            }
            else
            {
                A_s[threadIdx.y][threadIdx.x] = 0.0;
            }
            if (col < n && (i + threadIdx.y) < k)
            {
                B_s[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * n + col];
            }
            else
            {
                B_s[threadIdx.y][threadIdx.x] = 0.0;
            }
            // Synchronize threads
            __syncthreads();

            // Sum up values from A and B blocks
            for (j = 0; j < BLOCK_SIZE; ++j)
            {
                sum += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
            }
            // Synchronize threads
            __syncthreads();
        }
        // Keeping inside the boundaries
        if (row < m && col < n)
        {
            // Add the total value to C
            C[row * n + col] = sum;
        }
    }

    void matmult_gpu5(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Set grid and block size
        dim3 DimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Run kernel
        matmult_gpu5_kernel <<< DimGrid, DimBlock >>> (m, n, k, device_a, device_b, device_c);

        // Synchronize threads
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }

    void matmult_gpulib(int m, int n, int k, double *A, double *B, double *C)
    {
        // Declare and allocate memory on the device
        double *device_a, *device_b, *device_c, *trans_C;
        checkCudaErrors(cudaMalloc((void **) &device_a, m * k * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_b, k * n * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &device_c, m * n * sizeof(double)));
        // Allocate memory for temporary transposed C
        trans_C = (double *) malloc(m * n * sizeof(double));

        // Copy A and B from host to device
        checkCudaErrors(cudaMemcpy(device_a, A, m * k * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(device_b, B, k * n * sizeof(double), cudaMemcpyHostToDevice));

        // Create handle for the cuBlas library
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double alpha = 1.0, beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, device_a, k, device_b, n, &beta, device_c, m);
        cublasDestroy(handle);

        // Copy C from device to host
        checkCudaErrors(cudaMemcpy(trans_C, device_c, m * n * sizeof(double), cudaMemcpyDeviceToHost));

        // Transpose C back to row major format
        int row, col;
        for (row = 0; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                C[n * row + col] = trans_C[m * col + row];
            }
        }

        // Free allocated memory
        checkCudaErrors(cudaFree(device_a));
        checkCudaErrors(cudaFree(device_b));
        checkCudaErrors(cudaFree(device_c));
    }
}
