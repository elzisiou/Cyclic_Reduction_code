/*
* This is the cyclic reduction algorithm 
* cpu and gpu implementation
* results are checked for differences
* time is measured
* minavouronikoy@gmail.com
* elzisiou@gmail.com
* Master thesis
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

//#define float double

int log2(int x){
	return  (int)(log((float)x) / log(2.0));
}

__global__ void cr_forward(float *a, float *b, float *c, float *f, float *x,int size,int step_size,int i){
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int threadid = index_y * grid_width + index_x;

	int index1,index2,offset;
	float k1,k2;
	if(threadid>=step_size) return;
	int j = pow(2.0,i+1)*(threadid+1)-1;

	offset = pow(2.0,i);
	index1 = j-offset;
	index2 = j+offset;

	k1 = a[j]/b[index1];
	k2 = c[j]/b[index2];


	if(j == size - 1){
		k1 = a[j] / b[j-offset];
		b[j] = b[j] - c[j-offset] * k1;
		f[j] = f[j] - f[j-offset] * k1;
		a[j] = -a[j-offset] * k1;
		c[j] = 0.0;
	}
	else{
		k1 = a[j] / b[j-offset];
		k2 = c[j] / b[j+offset];
		b[j] = b[j] - c[j-offset] * k1 - a[j+offset] * k2;
		f[j] = f[j] - f[j-offset] * k1 - f[j+offset] * k2;
		a[j] = -a[j-offset] * k1;
		c[j] = -c[j+offset] * k2;
	}
}

__global__ void cr_backward(float *a, float *b, float *c, float *f, float *x,int size,int step_size,int i){
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int threadid = index_y * grid_width + index_x;

	int index1,index2,offset;
	if(threadid>=step_size) return;

	int j = pow(2.0,i+1)*(threadid+1)-1;



	offset = pow(2.0,i);
	index1 = j-offset;
	index2 = j+offset;

	if(j!=index1){
		if(index1 - offset < 0) x[index1] = (f[index1]- c[index1]*x[index1+offset])/b[index1]; 
		else x[index1] = (f[index1] - a[index1]*x[index1-offset] - c[index1]*x[index1+offset])/b[index1]; 
	}
	if(j!=index2){
		if(index2 + offset >=size) x[index2] = (f[index2] - a[index2]*x[index2-offset])/b[index2]; 
		else x[index2] = (f[index2] - a[index2]*x[index2-offset] - c[index2]*x[index2+offset])/b[index2]; 
	}
}

__global__ void cr_div(float *b,float *f,float *x,int index){
	x[index] = f[index]/b[index];
}

void cpu_cr(float *a,float *b,float *c,float *F,float *x,int size){
	int i,j,index1,index2,offset;
	float k1,k2;
	clock_t start,stop;
	start = clock();

	/*Part 1 - Forward Reduction */
	for(i=0;i<log2(size+1)-1;i++){
		for(j=pow(2.0,i+1)-1;j<size;j=j+pow(2.0,i+1)){
			offset = pow(2.0,i);
			index1 = j-offset;
			index2 = j+offset;

			k1 = a[j]/b[index1];
			k2 = c[j]/b[index2];

			if(j == size - 1){
				k1 = a[j] / b[j-offset];
				b[j] = b[j] - c[j-offset] * k1;
				F[j] = F[j] - F[j-offset] * k1;
				a[j] = -a[j-offset] * k1;
				c[j] = 0.0;
			}
			else{
				k1 = a[j] / b[j-offset];
				k2 = c[j] / b[j+offset];
				b[j] = b[j] - c[j-offset] * k1 - a[j+offset] * k2;
				F[j] = F[j] - F[j-offset] * k1 - F[j+offset] * k2;
				a[j] = -a[j-offset] * k1;
				c[j] = -c[j+offset] * k2;
			}
		}
	}



	/*part 2 - find the middle  */
	int index = (size-1)/2;
	x[index] = F[index]/b[index];

	/*part 3 - back substitution */
	for(i=log2(size+1)-2;i>=0;i--){
		for(j=pow(2.0,i+1)-1;j<size;j=j+pow(2.0,i+1)){
			offset = pow(2.0,i);
			index1 = j-offset;
			index2 = j+offset;


			if(j!=index1){
				if(index1 - offset < 0) x[index1] = (F[index1]- c[index1]*x[index1+offset])/b[index1]; 
				else x[index1] = (F[index1] - a[index1]*x[index1-offset] - c[index1]*x[index1+offset])/b[index1]; 
			}if(j!=index2){
				if(index2 + offset >=size) x[index2] = (F[index2] - a[index2]*x[index2-offset])/b[index2]; 
				else x[index2] = (F[index2] - a[index2]*x[index2-offset] - c[index2]*x[index2+offset])/b[index2]; 
			}

		}
	}
	stop = clock();
	printf("CPU Duration  %.2lf\n",(float)(stop-start)/CLOCKS_PER_SEC);
}

void printVector(float *vector,int size,char name){
	printf("Vector %c: \n",name);
	for(int i=0;i<size;i++){
		printf("%lf\n",vector[i]);
	}
}

void init_data(float **a,float **b,float **c,float **f,float **x,int size){
	int i,j;

	*a = (float *) malloc(sizeof(float)*size);
	*b = (float *) malloc(sizeof(float)*size);
	*c = (float *) malloc(sizeof(float)*size);
	*f = (float *) malloc(sizeof(float)*size);
	*x = (float *) malloc(sizeof(float)*size);

	for(i=0;i<size;i++) (*x)[i] = 0.0;
	for(j=0;j<size;j++) (*f)[j] = 15.0;

	for(i=0;i<size;i++){
		(*a)[i] = -1.0;
		(*b)[i] =  4.0;
		(*c)[i] = -1.0;
	}

	(*a)[0] = 0.0;
	(*c)[size-1]=0.0;
}

void diff_results(float *x,float *x_cpu,int size){
	int i;
	for(i=0;i<size;i++){
		if(fabs(x_cpu[i]-x[i])>0.1){
			printf("Diff %lf-%lf\n",x_cpu[i],x[i]);
			exit(1);
		}
	}

}
//calculate the block size according to size
void calc_dim(int size,dim3 *block,dim3 *grid){
	if(size<4)			{ block->x=1;block->y=1;}
	else if(size<16)	{ block->x=2;block->y=2;}
	else if(size<64)	{ block->x=4;block->y=4;}
	else if(size<256)	{ block->x=8;block->y=8;}
	else				{ block->x=16;block->y=16;}

	grid->x = (unsigned int)ceil(sqrt((double)size/block->x));
	grid->y = (unsigned int)ceil(sqrt((double)size/block->y));
}

int main(){
	clock_t start,stop;
	int i;
	int size = (int)pow(2.0,26)-1;

	float *a,*d_a; // lower diagonal vector
	float *b,*d_b; // main diagonal vector
	float *c,*d_c; //upper diagonal vector
	float *x,*d_x,*x_cpu;
	float *f,*d_f;

	dim3 dimBlock,dimGrid;

	cudaError_t error;

	a=b=c=x=f=NULL;
	d_a=d_b=d_c=d_x=d_f=NULL;

	init_data(&a,&b,&c,&f,&x,size);

	x_cpu = (float *)calloc(size,sizeof(float));
	// allocate device memory 
	cudaMalloc(&d_a,sizeof(float)*size);
	cudaMalloc(&d_b,sizeof(float)*size);
	cudaMalloc(&d_c,sizeof(float)*size);
	cudaMalloc(&d_x,sizeof(float)*size);
	cudaMalloc(&d_f,sizeof(float)*size);

	error = cudaGetLastError();
	if(error != cudaSuccess)  {    
		printf("CUDA malloc error: %s\n", cudaGetErrorString(error));   
		exit(0); 
	}
	//memory transfers
	cudaMemcpy(d_a,a,sizeof(float)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeof(float)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,c,sizeof(float)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_f,f,sizeof(float)*size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_x,x,sizeof(float)*size,cudaMemcpyHostToDevice);

	error = cudaGetLastError();
	if(error != cudaSuccess)  {    
		printf("CUDA memcpy error: %s\n", cudaGetErrorString(error));   
		exit(0); 
	}

	cudaFuncSetCacheConfig(cr_forward,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(cr_backward,cudaFuncCachePreferL1);


	int step_size;
	start = clock();

	for(i=0;i<log2(size+1)-1;i++){
		step_size = (size-pow(2.0,i+1))/pow(2.0,i+1)+1;
		calc_dim(step_size,&dimBlock,&dimGrid);

		cr_forward<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, d_f, d_x,size,step_size,i);
	}
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != cudaSuccess)  {    
		printf("CUDA forward kernel error: %s\n", cudaGetErrorString(error));   
		exit(0); 
	}

	cr_div<<<1,1>>>(d_b,d_f,d_x,(size-1)/2);
	cudaDeviceSynchronize();

	for(i=log2(size+1)-2;i>=0;i--){
		step_size = (size-pow(2.0,i+1))/pow(2.0,i+1)+1;
		calc_dim(step_size,&dimBlock,&dimGrid);
		cr_backward<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, d_f, d_x,size,step_size,i);
	}
	if(error != cudaSuccess)  {    
		printf("CUDA backward kernel error: %s\n", cudaGetErrorString(error));   
		exit(0); 
	}
	//copy back to host
	cudaMemcpy(x,d_x,sizeof(float)*size,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	stop = clock();

	if(error != cudaSuccess)  {    
		printf("CUDA memcpy x error: %s\n", cudaGetErrorString(error));   
		exit(0); 
	}
	cudaDeviceReset();

	printf("GPU Duration  %.2lf\n",(float)(stop-start)/CLOCKS_PER_SEC);
	cpu_cr(a,b,c,f,x_cpu,size);
	//check for differences
	diff_results(x,x_cpu,size);
}