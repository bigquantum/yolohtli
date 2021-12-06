
#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;

void __global__ singleCell_kernel(size_t pitch, stateVar g_out, 
	REAL *pt_d, int2 point) {

	pt_d[0] = g_out.u[point.x+nx_d*point.y];
	pt_d[1] = g_out.v[point.x+nx_d*point.y];

}

void singleCell_wrapper(size_t pitch, dim3 grid0D, dim3 block0D, stateVar gOut_d,
	int eSize, REAL *pt_h, REAL *pt_d, int2 point) {

	singleCell_kernel<<<grid0D,block0D>>>(pitch,gOut_d,pt_d,point);
	CudaCheckError();

	CudaSafeCall(cudaMemcpy(pt_h, pt_d, eSize*sizeof(REAL),cudaMemcpyDeviceToHost));

}

