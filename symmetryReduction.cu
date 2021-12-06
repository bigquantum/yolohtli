
//#include <stdio.h>
//#include <stdlib.h>

#include <iostream>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL invdx_d, invdy_d, hx_d, hy_d, Lx_d, Ly_d;
extern __constant__ bool solidSwitch_d;
extern __constant__ int tipOffsetX_d, tipOffsetY_d;
extern __constant__ float tipx0_d, tipy0_d;

__global__ void Cxy_field_kernel(advVar adv, REAL3 c, REAL3 phi, bool *solid) {

  /*------------------------------------------------------------------------
  * getting i and j global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

  /*------------------------------------------------------------------------
  * converting global index into matrix indices assuming
  * the column major structure of the matlab matrices
  *------------------------------------------------------------------------
  */

  const int i2d = i + j*nx_d;

  /*------------------------------------------------------------------------
  * Advection velocity matrix
  *------------------------------------------------------------------------
  */

  REAL x = (REAL)(i2d%nx_d);
  REAL y = (REAL)(floorf((i2d/nx_d)%nx_d));

  if ( solidSwitch_d ) {

    adv.x[i2d] = solid[i2d] ? hy_d*y*c.t - c.x*cos(phi.t) + c.y*sin(phi.t) : 0.0;
    adv.y[i2d] = solid[i2d] ? -hx_d*x*c.t - c.x*sin(phi.t) - c.y*cos(phi.t) : 0.0;

  } else {

    adv.x[i2d] = hy_d*y*c.t - c.x*cos(phi.t) + c.y*sin(phi.t);
    adv.y[i2d] = -hx_d*x*c.t - c.x*sin(phi.t) - c.y*cos(phi.t);

  }

}

}

void Cxy_field_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, 
  advVar adv, REAL3 c, REAL3 phi, bool *solid) {

  Cxy_field_kernel<<<grid2D, block2D>>>(adv, c, phi, solid);
  CudaCheckError();

}

__global__ void slice_kernel(stateVar g, sliceVar slice, sliceVar slice0,
	bool reduceSym, bool reduceSymStart, advVar adv, int scheme,
  bool *intglArea, int *tip_count, vec5dyn *tip_vector, int count) {

  /*------------------------------------------------------------------------
  * getting i and j global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  int cx, cy;

  if ( (i<nx_d) && (j<ny_d) ) {

  /*------------------------------------------------------------------------
  * Area where the derivatives are performed
  *------------------------------------------------------------------------
  */

  const int i2d = i + j*nx_d;

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

  REAL x = (int)i2d%nx_d;
  REAL y = floorf((i2d/nx_d)%nx_d);
  // bool sc = intglArea[i2d];

  int S = I2D(nx_d,i,coord_j(j-1));
  int N = I2D(nx_d,i,coord_j(j+1));
  int W = I2D(nx_d,coord_i(i-1),j);
  int E = I2D(nx_d,coord_i(i+1),j);

/*------------------------------------------------------------------------
* Slices
*------------------------------------------------------------------------
*/

switch (scheme) {

  case 1:

    if ( reduceSym ) {
  
      slice.ux[i2d] = convFB2ndOX(g.u,i,j,i2d,E,W,adv.x);
      slice.uy[i2d] = convFB2ndOY(g.u,i,j,i2d,N,S,adv.y);
      slice.vx[i2d] = convFB2ndOX(g.v,i,j,i2d,E,W,adv.x);
      slice.vy[i2d] = convFB2ndOY(g.v,i,j,i2d,N,S,adv.y);
  
  /*
      slice.ux[i2d] = convCentral2X(g.u,i,j,E,W);
      slice.uy[i2d] = convCentral2Y(g.u,i,j,N,S);
      slice.vx[i2d] = convCentral2X(g.v,i,j,E,W);
      slice.vy[i2d] = convCentral2Y(g.v,i,j,N,S);
  */

      slice.ut[i2d] = hx_d*x*slice.uy[i2d] - hy_d*y*slice.ux[i2d];
      slice.vt[i2d] = hx_d*x*slice.vy[i2d] - hy_d*y*slice.vx[i2d];

      if ( reduceSymStart ) {

        /*------------------------------------------------------------------------
        * Template (choose a frame of reference)
        *------------------------------------------------------------------------
        */

        slice0.ux[i2d] = slice.ux[i2d];
        slice0.uy[i2d] = slice.uy[i2d];
        slice0.vx[i2d] = slice.vx[i2d];
        slice0.vy[i2d] = slice.vy[i2d];

        slice0.ut[i2d] = hx_d*x*slice.uy[i2d] - hy_d*y*slice.ux[i2d];
        slice0.vt[i2d] = hx_d*x*slice.vy[i2d] - hy_d*y*slice.vx[i2d];

        reduceSymStart = false;

      }

    }

  break;

  case 2:

      if ( reduceSym ) {

        slice.ux[i2d] = sc ? convFB2ndOX(g.u,i,j,i2d,E,W,adv.x) : 0.0;
        slice.uy[i2d] = sc ? convFB2ndOY(g.u,i,j,i2d,N,S,adv.y) : 0.0;
        slice.vx[i2d] = sc ? convFB2ndOX(g.v,i,j,i2d,E,W,adv.x) : 0.0;
        slice.vy[i2d] = sc ? convFB2ndOY(g.v,i,j,i2d,N,S,adv.y) : 0.0;
      
        // slice.ux[i2d] = sc ? MUSCLx(g.u,i,j,i2d,W,E) : 0.0;
        // slice.uy[i2d] = sc ? MUSCLy(g.u,i,j,i2d,S,N) : 0.0;
        // slice.vx[i2d] = sc ? MUSCLx(g.v,i,j,i2d,W,E) : 0.0;
        // slice.vy[i2d] = sc ? MUSCLy(g.v,i,j,i2d,S,N) : 0.0;
          
        slice.ut[i2d] = sc ? hx_d*x*slice.uy[i2d] - hy_d*y*slice.ux[i2d] : 0.0;
        slice.vt[i2d] = sc ? hx_d*x*slice.vy[i2d] - hy_d*y*slice.vx[i2d] : 0.0;

        if ( reduceSymStart ) {

          /*------------------------------------------------------------------------
          * Template (choose a frame of reference)
          *------------------------------------------------------------------------
          */

          slice0.ux[i2d] = sc ? convCentral2X(g.u,i,j,E,W) : 0.0;
          slice0.uy[i2d] = sc ? convCentral2Y(g.u,i,j,N,S) : 0.0;
          slice0.vx[i2d] = sc ? convCentral2X(g.v,i,j,E,W) : 0.0;
          slice0.vy[i2d] = sc ? convCentral2Y(g.v,i,j,N,S) : 0.0;

          // slice0.ux[i2d] = sc ? MUSCLx(g.u,i,j,i2d,W,E) : 0.0;
          // slice0.uy[i2d] = sc ? MUSCLy(g.u,i,j,i2d,S,N) : 0.0;
          // slice0.vx[i2d] = sc ? MUSCLx(g.v,i,j,i2d,W,E) : 0.0;
          // slice0.vy[i2d] = sc ? MUSCLy(g.v,i,j,i2d,S,N) : 0.0;

          slice0.ut[i2d] = sc ? hx_d*x*slice0.uy[i2d] - hy_d*y*slice0.ux[i2d] : 0.0;
          slice0.vt[i2d] = sc ? hx_d*x*slice0.vy[i2d] - hy_d*y*slice0.vx[i2d] : 0.0;

          reduceSymStart = false;

          }

        }

  break;

  default:

    printf("No derivative scheme selected for slicing");

  break;

  }

}

}

__device__ REAL convCentral2X(REAL *f, const int i, const int j, int E, int W) {

  int WW = I2D(nx_d,coord_i(i-2),j);
  int EE = I2D(nx_d,coord_i(i+2),j);

  return (f[EE] - 8.0*f[E] + 8.0*f[W] - f[WW])*invdx_d*(1.0/6.0);

}

__device__ REAL convCentral2Y(REAL *f, const int i, const int j, int N, int S) {

  int SS = I2D(nx_d,i,coord_j(j-2));
  int NN = I2D(nx_d,i,coord_j(j+2));

  return (f[NN] - 8.0*f[N] + 8.0*f[S] - f[SS])*invdy_d*(1.0/6.0);

}

__device__ REAL convFB2ndOX(REAL *f, const int i, const int j, 
  int C, int E, int W, REAL *advx) {

  int WW = I2D(nx_d,coord_i(i-2),j);
  int EE = I2D(nx_d,coord_i(i+2),j);

  return (advx[C]>0.0) ? (-3.0*f[C] + 4.0*f[E] - f[EE])*invdx_d :
   (3.0*f[C] - 4.0*f[W] + f[WW])*invdx_d;

}

__device__ REAL convFB2ndOY(REAL *f, const int i, const int j, 
  int C, int N, int S, REAL *advy) {

  int SS = I2D(nx_d,i,coord_j(j-2));
  int NN = I2D(nx_d,i,coord_j(j+2));

  return (advy[C]>0.0) ? (-3.0*f[C] + 4.0*f[N] - f[NN])*invdy_d :
   (3.0*f[C] - 4.0*f[S] + f[SS])*invdy_d;

}

__device__ REAL convCentralX(REAL *f, int E, int W) {

  return (f[E] - f[W])*invdx_d;

}

__device__ REAL convCentralY(REAL *f, int N, int S) {

  return (f[N] - f[S])*invdy_d;

}

__device__ REAL MUSCLx(REAL *f, const int i, const int j, const int C, int W, int E) {

  // REAL BFD = (f[C] - f[W])/hx_d;
  // REAL FFD = (f[E] - f[C])/hx_d;

  int WW = I2D(nx_d,coord_i(i-2),j);
  int EE = I2D(nx_d,coord_i(i+2),j);

  // REAL BFD = (f[C] - f[WW])/invdx_d;
  // REAL FFD = (f[EE] - f[C])/invdx_d;

  REAL BFD = (3.0*f[C] - 4.0*f[W] + f[WW])*invdx_d;
  REAL FFD = (-3.0*f[C] + 4.0*f[E] - f[EE])*invdx_d;

  return ( (BFD>=0.0) && (FFD>=0.0) ) ? min(BFD,FFD) : ( ( (BFD<0.0) && (FFD<0.0) ) ? max(BFD,FFD) : 0.0 );

}

__device__ REAL MUSCLy(REAL *f, const int i, const int j, const int C, int S, int N) {

  // REAL BFD = (f[C] - f[S])/hy_d;
  // REAL FFD = (f[N] - f[C])/hy_d;

  int SS = I2D(nx_d,i,coord_j(j-2));
  int NN = I2D(nx_d,i,coord_j(j+2));

  // REAL BFD = (f[C] - f[SS])/invdy_d;
  // REAL FFD = (f[NN] - f[C])/invdy_d;

  REAL BFD = (3.0*f[C] - 4.0*f[S] + f[SS])*invdy_d;
  REAL FFD = (-3.0*f[C] + 4.0*f[N] - f[NN])*invdy_d;

  return ( (BFD>=0.0) && (FFD>=0.0) ) ? min(BFD,FFD) : ( ( (BFD<0.0) && (FFD<0.0) ) ? max(BFD,FFD) : 0.0 );

}

void slice_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, 
  stateVar g, sliceVar slice, sliceVar slice0,
	bool reduceSym, bool reduceSymStart, advVar adv, int scheme,
  bool *intglArea, int *tip_count, vec5dyn *tip_vector, int count) {

	slice_kernel<<<grid2D, block2D>>>(g,slice,slice0,
		reduceSym,reduceSymStart,adv,scheme,intglArea,
    tip_count,tip_vector,count);
	CudaCheckError();

}

/*------------------------------------------------------------------------
* Calculate drift velocities
*------------------------------------------------------------------------
*/

REAL3 solve_matrix(REAL3 c, REAL3 phi, REAL *Int) {

  REAL *x;
  x = vector(1,NSYM);
  
  // int *indx;
  // REAL **A,*x;

  // indx = ivector(1,NSYM);
  // A = matrix(1,NSYM,1,NSYM);

  // LU decomposition system solver

  // A[1][1] = Int[0]*cos(phi.t)+Int[1]*sin(phi.t); A[1][2] = Int[1]*cos(phi.t)-Int[0]*sin(phi.t); A[1][3] = Int[2];
  // A[2][1] = Int[3]*cos(phi.t)+Int[4]*sin(phi.t); A[2][2] = Int[4]*cos(phi.t)-Int[3]*sin(phi.t); A[2][3] = Int[5];
  // A[3][1] = Int[6]*cos(phi.t)+Int[7]*sin(phi.t); A[3][2] = Int[7]*cos(phi.t)-Int[6]*sin(phi.t); A[3][3] = Int[8];

  // x[1] = Int[9];
  // x[2] = Int[10];
  // x[3] = Int[11];

  // REAL d;

  /* Perform the decomposition */
  // ludcmp(A,indx,&d);

  /* Solve the linear system */
  // lubksb(A,indx,x);

  // c.x = x[1];
  // c.y = x[2];
  // c.t = x[3];

  // Check results

  // REAL partial;
  // for(int j=1;j<=3;j++) {
  //   partial = 0.0;
  //   for(int i=1;i<=3;i++) {
  //     partial += A[j][i]*x[i];
  //   }
  //   printf("Element %d = %12.16f. Error = %12.16f\n",j,partial,abs(partial-Int[j+8]));
  // }
  // printf("\n");

  // free_matrix(A,1,NSYM,1,NSYM);
  // free_ivector(indx,1,NSYM);
  // free_vector(x,1,NSYM);

/*
  printf("c.x = %12.16f, c.y = %12.16f, c.t = %12.16f\n",c.x,c.y,c.t);
  printf("phi.x = %12.16f, phi.y = %12.16f, phi.t = %12.16f\n",phi.x,phi.y,phi.t);
  printf("\n");
*/

  // Gauss-Jordan solver

  REAL a1, a2, a3, b1, b2, b3;
  REAL C1, C2, C3, d1, d2, d3;
  REAL b2p, b3p, c2p, c3p, c3pp, d2p, d3p, d3pp;

  a1 = Int[0]*cos(phi.t)+Int[1]*sin(phi.t); a2 = Int[1]*cos(phi.t)-Int[0]*sin(phi.t); a3 = Int[2];
  b1 = Int[3]*cos(phi.t)+Int[4]*sin(phi.t); b2 = Int[4]*cos(phi.t)-Int[3]*sin(phi.t); b3 = Int[5];
  C1 = Int[6]*cos(phi.t)+Int[7]*sin(phi.t); C2 = Int[7]*cos(phi.t)-Int[6]*sin(phi.t); C3 = Int[8];

  d1 = Int[9];
  d2 = Int[10];
  d3 = Int[11];

  b2p = a1/b1*b2-a2;
  b3p = a1/b1*b3-a3;
  d2p = a1/b1*d2-d1;

  c2p = a1/C1*C2-a2;
  c3p = a1/C1*C3-a3;
  d3p = a1/C1*d3-d1;

  c3pp = b2p/c2p*c3p-b3p;
  d3pp = b2p/c2p*d3p-d2p;

  // Back substitution
  x[3] = d3pp/c3pp;
  x[2] = (d2p-b3p*x[3])/b2p;
  x[1] = (d1-a2*x[2]-a3*x[3])/a1;

  c.x = x[1];
  c.y = x[2];
  c.t = x[3];
  
  return c;

}
