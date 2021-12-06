
#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL hx_d, hy_d;
extern __constant__ int tipOffsetX_d, tipOffsetY_d;
extern __constant__ float tipx0_d, tipy0_d;

__global__ void trapz_kernel(REAL *f, REAL *g, REAL *h, REAL *w,
  REAL *dot, REAL *coeffTrapz, int *tip_count, vec5dyn *tip_vector,
  int count) {

  unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int stride = blockDim.x*gridDim.x;

  __shared__ REAL cache[BLOCKSIZE_1D];

  REAL temp = 0.0;

  while(index < nx_d*ny_d) {

  //   // coeffTrapz defines a circular area of coefficients
  //   temp += (f[index]*g[index] + 
  //            h[index]*w[index] )*coeffTrapz[index];

  int i = (int)index%nx_d;
  int j = (int)floorf((index/nx_d)%nx_d);
  int cx, cy;
  REAL fxg;

  int ic = i-nx_d/2;
  int jc = j-ny_d/2;

  if ( count == 0 ) {
    cx = __float2int_rn(tipx0_d-nx_d/2);
    cy = __float2int_rn(tipy0_d-ny_d/2);
  } else {
    cx = __float2int_rn(tip_vector[*tip_count-1].x-nx_d/2);
    cy = __float2int_rn(tip_vector[*tip_count-1].y-ny_d/2);
  }

  bool sc = ( (ic-cx)*(ic-cx) + (jc-cy)*(jc-cy) ) 
    < tipOffsetX_d*tipOffsetY_d ? true : false;
  bool scb = ( (ic-cx)*(ic-cx) + (jc-cy)*(jc-cy) ) 
    == tipOffsetX_d*tipOffsetY_d ? true : false;

  fxg = sc ? (f[index]*g[index] + h[index]*w[index]) : 0.0 ; 

  temp += sc ? 4.0*fxg : ( scb ? fxg : 2.0*fxg );

    index += stride;


  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  // reduction
  unsigned int i = blockDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      cache[threadIdx.x] += cache[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    atomicAdd(dot, 0.25*hx_d*hy_d*cache[0]);
  }
  
}


void trapz_wrapper(dim3 grid1D, dim3 block1D, sliceVar slice, sliceVar slice0, 
  stateVar velTan, REAL *integrals, REAL *coeffTrapz,
  int *tip_count, vec5dyn *tip_vector, int count) {

  REAL *prod;
  REAL *prod_d;
  prod = (REAL*)malloc(sizeof(REAL));
  CudaSafeCall(cudaMalloc((void**)&prod_d, sizeof(REAL)));

  // It's important to restart the output vector
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ux, slice.ux, slice0.vx, slice.vx,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[0] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ux, slice.uy, slice0.vx, slice.vy,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[1] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ux, slice.ut, slice0.vx, slice.vt, 
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[2] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.uy, slice.ux, slice0.vy, slice.vx, 
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[3] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.uy, slice.uy, slice0.vy, slice.vy,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[4] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.uy, slice.ut, slice0.vy, slice.vt,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[5] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ut, slice.ux, slice0.vt, slice.vx,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[6] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ut, slice.uy, slice0.vt, slice.vy,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[7] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ut, slice.ut, slice0.vt, slice.vt, 
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[8] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ux, velTan.u, slice0.vx, velTan.v,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[9] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.uy, velTan.u, slice0.vy, velTan.v,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[10] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  trapz_kernel<<<grid1D, block1D>>>(slice0.ut, velTan.u, slice0.vt, velTan.v,
    prod_d,coeffTrapz,tip_count,tip_vector, count);
  CudaCheckError();
  CudaSafeCall(cudaMemcpy(prod, prod_d, sizeof(REAL), cudaMemcpyDeviceToHost));
  integrals[11] = *prod;
  CudaSafeCall(cudaMemset(prod_d, 0.0, sizeof(REAL)));

  free(prod);
  CudaSafeCall(cudaFree(prod_d));

}
