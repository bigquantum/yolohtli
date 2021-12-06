#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>

#include "./common/SOIL.h"

#include "typeDefinition.cuh"
// #include "globalVariables.cuh"

#include "hostPrototypes.h"
#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __device__ vec5dyn tip_vector[TIPVECSIZE];
extern __device__ int tip_count;
extern paramVar param;

/*------------------------------------------------------------------------
* Set optimal block and grid sizes for the CUDA kernel
*------------------------------------------------------------------------
*/

__host__ __device__ int iDivUp(int a, int b) {
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

/*------------------------------------------------------------------------
* Filter algorithms
*------------------------------------------------------------------------
*/

__device__ void push_back3(float3 pt, int *count, float3 *count_vector) {

  count_vector[atomicAdd(count, 1)] = pt;

}

__device__ void push_back5(vec5dyn pt, int *count, vec5dyn *count_vector) {

  count_vector[atomicAdd(count, 1)] = pt;

}

__device__ void plot_field(float ptx, float pty, bool *plot_array) {

  int xIdx = floor(ptx);
  int yIdx = floor(pty);
  plot_array[I2D(nx_d,xIdx,yIdx)] = true;

}

/*------------------------------------------------------------------------
* Compare two quantities
*------------------------------------------------------------------------
*/

__device__ bool equals( REAL a, REAL b, REAL tolerance ) {
    return ( a == b ) ||
      ( ( a <= ( b + tolerance ) ) &&
        ( a >= ( b - tolerance ) ) );
}

/*------------------------------------------------------------------------
* Indices for finite differences
*------------------------------------------------------------------------
*/

__device__ int coord_i(int i) {

  return (int)((i>=0) && (i<nx_d))*i + (int)(i<0)*(-i) + (int)(i>=nx_d)*(2*(nx_d-1)-i);

}

__device__ int coord_j(int j) {

  return (int)((j>=0) && (j<ny_d))*j + (int)(j<0)*(-j) + (int)(j>=ny_d)*(2*(ny_d-1)-j);

}

/*------------------------------------------------------------------------
* Print last tip trajectory points
*------------------------------------------------------------------------
*/

void saveTipLast(int *tip_count, vec5dyn *tip_vector, paramVar *param) {

  int *tip_pts;
  tip_pts = (int*)malloc(sizeof(int));
  CudaSafeCall(cudaMemcpy(tip_pts,tip_count,sizeof(int),cudaMemcpyDeviceToHost));

  if (*tip_pts > TIPVECSIZE ) {
    printf("ERROR: NUMBER OF TIP POINTS EXCEEDS tip_vector SIZE\n");
    exit(0);
  }

  vec5dyn *tip_array;
  tip_array = (vec5dyn*)malloc((*tip_pts)*sizeof(vec5dyn));
  CudaSafeCall(cudaMemcpy(tip_array,tip_vector,(*tip_pts)*sizeof(vec5dyn),cudaMemcpyDeviceToHost));

  if ( *tip_pts > 0 ) {
    // Record last tip point
    param->tipx = tip_array[(*tip_pts)-1].x;
    param->tipy = tip_array[(*tip_pts)-1].y;
  } else {
    param->tipx = -1.0f;
    param->tipy = -1.0f;
  }

  free(tip_pts);
  free(tip_array);

}

/*------------------------------------------------------------------------
* Linear interpolation
*------------------------------------------------------------------------
*/

float host_lerp(float v0, float v1, float t) {
  return (1 - t) * v0 + t * v1;
}

__device__ inline REAL my_lerp(REAL v0, REAL v1, REAL t) {
    //return (1.f-t)*v0 + t*v1;
    return fma(t, v1, fma(-t, v0, v0));
}

/*------------------------------------------------------------------------
* Swap array pointers
*------------------------------------------------------------------------
*/

void swap(float* &a, float* &b) {
  float *temp = a;
  a = b;
  b = temp;
}

void swapSoA(stateVar *A, stateVar *B) {
    stateVar temp = *A;
    *A = *B;
    *B = temp;
}

/*------------------------------------------------------------------------
* Sign function
*------------------------------------------------------------------------
*/

__device__ int sign(REAL x) { 

  int t = x < 0.0 ? -1 : 0;

  return x > 0.0 ? 1 : t;

}

/*------------------------------------------------------------------------
* Take screenshot
*------------------------------------------------------------------------
*/

void screenShot(int w, int h) {
	
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);

    char name[100];
    sprintf(name, "./DATA/screenshots/figure_%d-%d-%d_%d-%d-%d.bmp", 
    	tm.tm_year + 1900, tm.tm_mon + 1, 
    	tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
      /* save a screenshot */
      // sudo apt-get install libsoil-dev
    SOIL_save_screenshot(name,
                         SOIL_SAVE_TYPE_BMP,
                         0, 0, w, h);

}

/*------------------------------------------------------------------------
* Press Enter key to confirm
*------------------------------------------------------------------------
*/

void pressEnterKey(void) {
  // Ask for ENTER key
  printf("Press [Enter] key to continue\n");
  printf("[Ctrl]+[C] to terminate program.\n");
  while(getchar()!='\n'); // option TWO to clean stdin
  getchar(); // wait for ENTER
}

/*------------------------------------------------------------------------
* Conduction block
*------------------------------------------------------------------------
*/

void conductionBlock(int memSize, bool counterclock, bool clock1,
  stateVar g_h, stateVar g_present_d) {

  int i, j, idx;

  CudaSafeCall(cudaMemcpy(g_h.u, g_present_d.u, memSize,
    cudaMemcpyDeviceToHost));

  if (counterclock) {
    for (j=0;j<(param.ny/2+70);j++) {
    // for (j=0;j<(param.ny/2-10);j++) {
    // for (j=0;j<(param.ny/2+50);j++) {
      for (i=0;i<(param.nx);i++) {
        idx = i + param.nx * j;
        g_h.u[idx] = 0.0;
        }
      }

    }

  if (clock1) {

    for (j=0;j<param.ny;j++) {
      for (i=(param.nx/2);i<(param.nx);i++) {
        idx = i + param.nx * j;
        g_h.u[idx] = 0.0;
        }
      }

    }

  CudaSafeCall(cudaMemcpy(g_present_d.u, g_h.u, memSize,
    cudaMemcpyHostToDevice));

}

/*------------------------------------------------------------------------
* Checks for the voltage level
*------------------------------------------------------------------------
*/

bool isThereFib(REAL *voltage, paramVar param) {

  REAL v = 0.0;
  for (int i=floor(9*param.wnx/10.0);i<param.wnx;i++) {
    v += voltage[i];
  }

  return v>0.5 ? true : false;

}