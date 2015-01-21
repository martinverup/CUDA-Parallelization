#include <stdio.h>
#include <helper_cuda.h> 


void init(int N, double delta, double *U, double *U_old, double *F) {

	int temp_N = N + 2; //the boundries
	// Declare relative coordinates
	double x = -1.0;
	double y = -1.0;
	double x_lower = 0.0;
	double x_upper = 1.0 / 3.0;
	double y_lower = -2.0 / 3.0;
	double y_upper = -1.0 / 3.0;
	int i, j;
	for (i = 0; i < temp_N; i++)
	{
		for (j = 0; j < temp_N; j++)
		{
            	F[i * (temp_N)+j] = 0.0;
            	U[i * (temp_N)+j] = 0.0;
            	U_old[i * (temp_N)+j] = 0.0;
		    // Place radiator for F in the right place
		    if (x <= x_upper && x >= x_lower && y <= y_upper && y >= y_lower)
		    {
			// Set radiator value to 200 degrees
			F[i * temp_N + j] = 200.0;
		    }
		    // Place temperature for walls
		    if (i == (temp_N - 1) || i == 0 || j == (temp_N - 1))
		    {
		        // Set temperature to 20 degrees for 3 of the walls
		        U[i *(temp_N) + j]= 20.0;
		        U_old[i * (temp_N) + j] = 20.0;
		    }	
		    // Move relative coordinates by one unit of grid spacing
		    y += delta;
		}
	// Move relative coordinates by one unit of grid spacing
	x += delta;
	y = -1.0;
	}
}

__global__ void jacobi(int N, double delta2, double *U, double *U_old, double *F) {
        int new_N = (N+2);  
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int j;
	for (j = 1; j < new_N-1; j++)
	    {
	        // Calculate new value from surrounding points
	        U_old[i * new_N + j] = (U[i * new_N+ (j-1)] + U[i * new_N + (j+1)] + U[(i-1) * new_N + j] + U[(i + 1) * new_N + j] + (delta2 * F[i * new_N + j])) * 0.25;
	    }
	__syncthreads();
}

void print_matrix(int N, double *M)
{
	int temp_N = N + 2;
	int i, j;
	for (i = temp_N - 1; i >= 0; i--)
	{
	for (j = 0; j < temp_N; j++)
	{
	    // Swap indecies to show correct x and y-axes
	    printf("%.2f\t", M[j * temp_N + i]);
	}
	printf("\n");
	}
}

int main() {

	int N = 32;
	int k = 1000;
	int size = (N+2) * (N+2) * sizeof(double);
	double delta = 2.0 / ((double) N - 1.0);
	double delta2 = delta * delta;
	
	dim3 DimBlock(1);
	dim3 DimGrid(N);
	
	double *U_dev;
	double *U_old_dev;
	double *F_dev;
	double *temp;
	double *U_host;
	double *U_old_host;
	double *F_host;
	
	//alloctating memory on host
	U_host = (double *) malloc(size);
	U_old_host = (double *) malloc(size);
	F_host = (double *) malloc(size);
	
	//initializing the arrays
	init(N, delta, U_host, U_old_host, F_host);
	
	//allocating memory on device
	checkCudaErrors(cudaMalloc((void**) &U_dev,size));
	checkCudaErrors(cudaMalloc((void**) &U_old_dev,size));
	checkCudaErrors(cudaMalloc((void**) &F_dev, size));
	
	//copying memory from CPU to GPU
	checkCudaErrors(cudaMemcpy(U_dev, U_host, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(U_old_dev, U_old_host, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(F_dev, F_host, size, cudaMemcpyHostToDevice));
	int h;
	for(h = 0; h < k; h++) {
	jacobi<<<DimGrid, DimBlock>>>(N, delta2, U_dev, U_old_dev, F_dev);
		//swapping pointers
        	temp = U_dev;
                U_dev = U_old_dev;
                U_old_dev = temp;
	}
	checkCudaErrors(cudaMemcpy(U_host, U_dev, size, cudaMemcpyDeviceToHost));
	
	print_matrix(N, U_host);
	//freeing the memory in the end
	free(U_host);
	free(U_old_host);
	free(F_host);
	checkCudaErrors(cudaFree(U_dev));
	checkCudaErrors(cudaFree(U_old_dev));
	checkCudaErrors(cudaFree(F_dev));
	return 0;
}
