
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL invdx_d, invdy_d;
extern __constant__ REAL dt_d;
extern __constant__ REAL Uth_d;
extern __constant__ bool solidSwitch_d, tipGrad_d;
extern __constant__ int tipOffsetX_d, tipOffsetY_d;

__global__ void spiralTip_kernel(size_t pitch, REAL *g_past,
  REAL *g_present, bool *tip_plot, int *tip_count, vec5dyn *tip_vector,
  REAL physicalTime) {

  /*------------------------------------------------------------------------
  * Getting i, and k global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  int ic, jc;
  int s0, sx, sy, sxy;

  REAL x1, x2, x3, x4, y1, y2, y3, y4;
  REAL x3y1, x4y1, x3y2, x4y2, x1y3, x2y3, x1y4, x2y4, x2y1, x1y2, x4y3, x3y4;
  REAL den1, den2, ctn1, ctn2, disc, px, py;
  float2 tip;

  /*------------------------------------------------------------------------
  * Return if we are outside the domain
  *------------------------------------------------------------------------
  */

  if ( (i<nx_d) && (j<ny_d) ) {

    if ( solidSwitch_d ) {

      ic = i-nx_d/2;
      jc = j-ny_d/2;
      if ( (ic*ic + jc*jc) < tipOffsetX_d*tipOffsetY_d ) {

        s0   = I2D(nx_d,i,j);
        sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
        sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
        sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

        /*------------------------------------------------------------------------
        * XY plane
        *------------------------------------------------------------------------
        */

        x1 = g_present[s0];
        x2 = g_present[sx];
        x4 = g_present[sy];
        x3 = g_present[sxy];

        y1 = g_past[s0];
        y2 = g_past[sx];
        y4 = g_past[sy];
        y3 = g_past[sxy];

        x3y1 = x3*y1;
        x4y1 = x4*y1;
        x3y2 = x3*y2;
        x4y2 = x4*y2;
        x1y3 = x1*y3;
        x2y3 = x2*y3;
        x1y4 = x1*y4;
        x2y4 = x2*y4;
        x2y1 = x2*y1;
        x1y2 = x1*y2;
        x4y3 = x4*y3;
        x3y4 = x3*y4;

        den1 = 2.0*(x3y1 - x4y1 - x3y2 + x4y2 - x1y3 + x2y3 + x1y4 - x2y4);
        den2 = 2.0*(x2y1 - x3y1 - x1y2 + x4y2 + x1y3 - x4y3 - x2y4 + x3y4);

        ctn1 = x1 - x2 + x3 - x4 - y1 + y2 - y3 + y4;
        ctn2 = x3y1 - 2.0*x4y1 + x4y2 - x1y3 + 2.0*x1y4 - x2y4;

        disc = sqrt(4.0 * ( x3y1 - x3y2 - x4y1 + x4y2 - x1y3 + x1y4 + x2y3 - x2y4 )
          * (x4y1 - x1y4 + Uth_d * (x1 - x4 - y1 + y4)) +
          ( -ctn2 + Uth_d * ctn1 ) * (-ctn2 + Uth_d * ctn1 ));

        px = ctn2 -Uth_d * ctn1;
        py = Uth_d * ctn1 - x3y1 + x4y2 + x1y3 - x2y4 + 2.0*(x2y1 - x1y2);

        /*------------------------------------------------------------------------
        * XY plane
        * Clockwise direction tip
        *------------------------------------------------------------------------
        */

        tip.x = (px + disc)/den1;
        tip.y = (py + disc)/den2;

        tipRecordPlane(i,j,tip,physicalTime,g_present,
          tip_plot,tip_count,tip_vector);
        
        /*------------------------------------------------------------------------
        * Counterclockwise direction tip
        *------------------------------------------------------------------------
        */

        tip.x = (px - disc)/den1;
        tip.y = (py - disc)/den2;

        tipRecordPlane(i,j,tip,physicalTime,g_present,
          tip_plot,tip_count,tip_vector);

      }

    } else { // Square domain
  
      if ( (i>=1) && (i<(nx_d-2)) && (j>=1) && (j<(ny_d-2)) ) {
      // if ( (i>=(nx_d/2-tipOffsetX_d)) && (i<(nx_d/2+tipOffsetX_d)) &&
      //   (j>=(ny_d/2-tipOffsetY_d)) && (j<(ny_d/2+tipOffsetY_d)) ) {

        s0   = I2D(nx_d,i,j);
        sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
        sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
        sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

        /*------------------------------------------------------------------------
        * XY plane
        *------------------------------------------------------------------------
        */

        x1 = g_present[s0];
        x2 = g_present[sx];
        x4 = g_present[sy];
        x3 = g_present[sxy];

        y1 = g_past[s0];
        y2 = g_past[sx];
        y4 = g_past[sy];
        y3 = g_past[sxy];

        x3y1 = x3*y1;
        x4y1 = x4*y1;
        x3y2 = x3*y2;
        x4y2 = x4*y2;
        x1y3 = x1*y3;
        x2y3 = x2*y3;
        x1y4 = x1*y4;
        x2y4 = x2*y4;
        x2y1 = x2*y1;
        x1y2 = x1*y2;
        x4y3 = x4*y3;
        x3y4 = x3*y4;

        den1 = 2.0*(x3y1 - x4y1 - x3y2 + x4y2 - x1y3 + x2y3 + x1y4 - x2y4);
        den2 = 2.0*(x2y1 - x3y1 - x1y2 + x4y2 + x1y3 - x4y3 - x2y4 + x3y4);

        ctn1 = x1 - x2 + x3 - x4 - y1 + y2 - y3 + y4;
        ctn2 = x3y1 - 2.0*x4y1 + x4y2 - x1y3 + 2.0*x1y4 - x2y4;

        disc = sqrt(4.0 * ( x3y1 - x3y2 - x4y1 + x4y2 - x1y3 + x1y4 + x2y3 - x2y4 )
          * (x4y1 - x1y4 + Uth_d * (x1 - x4 - y1 + y4)) +
          ( -ctn2 + Uth_d * ctn1 ) * (-ctn2 + Uth_d * ctn1 ));

        px = ctn2 -Uth_d * ctn1;
        py = Uth_d * ctn1 - x3y1 + x4y2 + x1y3 - x2y4 + 2.0*(x2y1 - x1y2);

        /*------------------------------------------------------------------------
        * XY plane
        * Clockwise direction tip
        *------------------------------------------------------------------------
        */

        tip.x = (px + disc)/den1;
        tip.y = (py + disc)/den2;

        if (disc>=0.0 ) {
        tipRecordPlane(i,j,tip,physicalTime,g_present,
          tip_plot,tip_count,tip_vector);
        }
        /*------------------------------------------------------------------------
        * Counterclockwise direction tip
        *------------------------------------------------------------------------
        */

        tip.x = (px - disc)/den1;
        tip.y = (py - disc)/den2;

        if (disc>=0.0 ) {
        tipRecordPlane(i,j,tip,physicalTime,g_present,
          tip_plot,tip_count,tip_vector);
        }

      } 

    } // Square ends

  }

}

__device__ void tipRecordPlane(const int i, const int j, float2 tip,
  REAL pTime, REAL *g_p, bool *tip_plot, int *tip_count, vec5dyn *tip_vector) {

  float2 g;

  if ( ( ((tip.x > 0.0) && (tip.x < 1.0)) && ((tip.y > 0.0) && (tip.y < 1.0)) )) {

    /*------------------------------------------------------------------------
    * Calculate the gradient
    *------------------------------------------------------------------------
    */

    if ( tipGrad_d ) {
      g = gradient(i,j,tip.x,tip.y,g_p);
    } else {
      g = make_float2(0.0,0.0);
    }

    /*------------------------------------------------------------------------
    * Save data
    *------------------------------------------------------------------------
    */

    vec5dyn data = { .x = (float)(i+tip.x), .y = (float)(j+tip.y),
     .vx = (float)g.x, .vy = (float)g.y, .t = (float)pTime };

    push_back5(data,tip_count,tip_vector);
    plot_field(data.x,data.y,tip_plot);

  }

}


__global__ void spiralTipNewton_kernel(size_t pitch, REAL *g_past,
  REAL *g_present, bool *tip_plot, int *tip_count, vec5dyn *tip_vector,
  REAL physicalTime) {

  /*------------------------------------------------------------------------
  * Getting i, and k global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  int ic, jc;
  int s0, sx, sy, sxy;
  float2 tip;

  if ( (i<nx_d) && (j<ny_d) ) {

    if ( solidSwitch_d ) {

      ic = i-nx_d/2;
      jc = j-ny_d/2;
      if ( (ic*ic + jc*jc) < tipOffsetX_d*tipOffsetY_d ) {

        s0   = I2D(nx_d,i,j);
        sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
        sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
        sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

        REAL px1, px2, px3, px4, py1, py2, py3, py4;

        /*------------------------------------------------------------------------
        * XY plane
        *------------------------------------------------------------------------
        */

        px1 = g_present[s0];
        px2 = g_present[sx];
        px4 = g_present[sy];
        px3 = g_present[sxy];

        py1 = g_past[s0];
        py2 = g_past[sx];
        py4 = g_past[sy];
        py3 = g_past[sxy];

        REAL r1, r2, J11, J12, J21, J22, detJ, u1, u2;
        REAL s = 0.5;
        REAL t = 0.5;
        int maxIter = 4;

        int k;

        /*------------------------------------------------------------------------
        * Newton method
        *------------------------------------------------------------------------
        */

        for (k=0;k<maxIter;k++) {

          r1 = px1*(1.0-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t - Uth_d; //residual
          r2 = py1*(1.0-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t - Uth_d; //residual
          J11 = -px1*(1.0-t) + px2*(1.0-t) + px3*t - px4*t; // dr1/ds
          J21 = -py1*(1.0-t) + py2*(1.0-t) + py3*t - py4*t; // dr2/ds
          J12 = -px1*(1.0-s) - px2*s + px3*s + px4*(1.0-s); // dr1/dt
          J22 = -py1*(1.0-s) - py2*s + py3*s + py4*(1.0-s); // dr2/dt

          detJ = J11*J22 - J12*J21;

          if ( !equals( detJ, 0.0, 1e-14 ) ) {

            s -= (J22*r1 - J12*r2)/detJ;
            t -= (-J21*r1 + J11*r2)/detJ;
            s = min(max(s, 0.0),1.0);
            t = min(max(t, 0.0),1.0);

          } else {
            s = -1.0; // Arbitrary values
            t = -1.0;
          }

        }

        // Set the negative condition to a value you are not looking for.
        u1 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
          px1*(1-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t : 0.0;
        u2 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
          py1*(1-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t : 0.0;

        if ( equals( u1, Uth_d, 1e-15 ) && equals( u2, Uth_d, 1e-15 ) ) {

          tip.x = (float)s;
          tip.y = (float)t;

          tipRecordPlane(i,j,tip,physicalTime,g_present,
            tip_plot,tip_count,tip_vector);

        }

      }

    } else { // Square domain

      if ( (i>=(nx_d/2-tipOffsetX_d)) && (i<(nx_d/2+tipOffsetX_d)) &&
        (j>=(ny_d/2-tipOffsetY_d)) && (j<(ny_d/2+tipOffsetY_d)) ) {

        s0   = I2D(nx_d,i,j);
        sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
        sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
        sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

        REAL px1, px2, px3, px4, py1, py2, py3, py4;

        /*------------------------------------------------------------------------
        * XY plane
        *------------------------------------------------------------------------
        */

        px1 = g_present[s0];
        px2 = g_present[sx];
        px4 = g_present[sy];
        px3 = g_present[sxy];

        py1 = g_past[s0];
        py2 = g_past[sx];
        py4 = g_past[sy];
        py3 = g_past[sxy];

        REAL r1, r2, J11, J12, J21, J22, detJ, u1, u2;
        REAL s = 0.5;
        REAL t = 0.5;
        int maxIter = 4;

        int k;

        /*------------------------------------------------------------------------
        * Newton method
        *------------------------------------------------------------------------
        */

        for (k=0;k<maxIter;k++) {

          r1 = px1*(1.0-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t - Uth_d; //residual
          r2 = py1*(1.0-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t - Uth_d; //residual
          J11 = -px1*(1.0-t) + px2*(1.0-t) + px3*t - px4*t; // dr1/ds
          J21 = -py1*(1.0-t) + py2*(1.0-t) + py3*t - py4*t; // dr2/ds
          J12 = -px1*(1.0-s) - px2*s + px3*s + px4*(1.0-s); // dr1/dt
          J22 = -py1*(1.0-s) - py2*s + py3*s + py4*(1.0-s); // dr2/dt

          detJ = J11*J22 - J12*J21;

          if ( !equals( detJ, 0.0, 1e-14 ) ) {

            s -= (J22*r1 - J12*r2)/detJ;
            t -= (-J21*r1 + J11*r2)/detJ;
            s = min(max(s, 0.0),1.0);
            t = min(max(t, 0.0),1.0);

          } else {
            s = -1.0; // Arbitrary values
            t = -1.0;
          }

        }

        // Set the negative condition to a value you are not looking for.
        u1 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
          px1*(1-s)*(1.0-t) + px2*s*(1.0-t) + px3*s*t + px4*(1.0-s)*t : 0.0;
        u2 = (s>=0.0) && (s<=1.0) && (t>=0.0) && (t<=1.0) ? 
          py1*(1-s)*(1.0-t) + py2*s*(1.0-t) + py3*s*t + py4*(1.0-s)*t : 0.0;

        if ( equals( u1, Uth_d, 1e-15 ) && equals( u2, Uth_d, 1e-15 ) ) {

          tip.x = (float)s;
          tip.y = (float)t;

          tipRecordPlane(i,j,tip,physicalTime,g_present,
            tip_plot,tip_count,tip_vector);


        }

      }

    }

  }

}

__global__ void abouzarTip_kernel(size_t pitch, REAL *g_past,
  REAL *g_present, bool *tip_plot, int *tip_count, vec5dyn *tip_vector,
  REAL physicalTime) {

  /*------------------------------------------------------------------------
  * Getting i, and k global indices
  *------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  int s0, sx, sy, sxy;


  /*------------------------------------------------------------------------
  * Return if we are outside the domain
  *------------------------------------------------------------------------
  */

  if ( (i<nx_d) && (j<ny_d) ) {

    s0   = I2D(nx_d,i,j);
    sx   = ( i<(nx_d-1)  ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
    sy   = ( j<(ny_d-1)  ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);
    sxy  = ( (j<(ny_d-1)) && (i<(nx_d-1) ) ) ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i,j);

    // if ( (sx<(nx_d-1)) && (sy<(ny_d-1)) ) {

    tip_plot[s0] += abubuFilament(s0,sx,sy,sxy,g_past,g_present);

    // }

    }


}

/*------------------------------------------------------------------------
* Abouzar's algorithm for  the tip trajectory
*------------------------------------------------------------------------
*/

__device__ bool abubuFilament(int s0, int sx, int sy,  int sxy, 
    REAL *g_past, REAL *g_present) {

  REAL v0, vx, vy, vxy;
  REAL d0, dx, dy, dxy;
  REAL f0, fx, fy, fxy;
  REAL s;
  bool bv, bdv;

  v0   = g_present[s0];
  vx   = g_present[sx];
  vy   = g_present[sy];
  vxy  = g_present[sxy];

  f0   = v0   - Uth_d;
  fx   = vx   - Uth_d;
  fy   = vy   - Uth_d;
  fxy  = vxy  - Uth_d;

  s = STEP(0.f, f0  )
    + STEP(0.f, fx  )
    + STEP(0.f, fy  )
    + STEP(0.f, fxy );

  bv = ( s>0.5f ) && ( s<3.5f );

  d0   = v0   - g_past[s0];
  dx   = vx   - g_past[sx];
  dy   = vy   - g_past[sy];
  dxy  = vxy  - g_past[sxy];

  s = STEP(0.f, d0  )
    + STEP(0.f, dx  )
    + STEP(0.f, dy  )
    + STEP(0.f, dxy );

  bdv = ( s>0.5f ) && ( s<3.5f );

  return ( bdv && bv );

}

__device__ float2 gradient(const int i, const int j, REAL s, REAL t, REAL *g_p) {

  float2 g;
  REAL gx1, gx2, gx3, gx4, gy1, gy2, gy3, gy4;

  /*------------------------------------------------------------------------
  * Gradient indices
  *------------------------------------------------------------------------
  */

  int S = ( j>0 )                  ? I2D(nx_d,i,j-1) : I2D(nx_d,i,j+1) ;
  int Sx = ( (j>0) && (i<(nx_d-1)) ) ? I2D(nx_d,i+1,j-1) : I2D(nx_d,i-1,j+1) ;
  int Sy =                           I2D(nx_d,i,j) ;
  int Sxy = ( i<(nx_d-1) )           ? I2D(nx_d,i+1,j) : I2D(nx_d,i-1,j) ;

  int N = ( j<(ny_d-1)  )                  ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j-1) ;
  int Nx = ( (i<(nx_d-1)) && (j<(ny_d-1)) )  ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i-1,j-1) ;
  int Ny = ( j<(ny_d-2) )                  ? I2D(nx_d,i,j+2) : (( j==(ny_d-2) ) ? I2D(nx_d,i,j) : I2D(nx_d,i,j-1)) ;
  int Nxy = ( (i<(nx_d-1)) && (j<(ny_d-2)) ) ? I2D(nx_d,i+1,j+2) : ((j==(ny_d-2)) ?  I2D(nx_d,i-1,j) : I2D(nx_d,i-1,j-1)) ;

  int W = ( i>0 )                  ? I2D(nx_d,i-1,j) : I2D(nx_d,i-1,j) ;
  int Wx =                           I2D(nx_d,i,j) ;
  int Wy = ( (i>0) && (j<(ny_d-1)) ) ? I2D(nx_d,i-1,j+1) : I2D(nx_d,i+1,j-1) ;
  int Wxy = ( (j<(ny_d-1)) )         ? I2D(nx_d,i,j+1) : I2D(nx_d,i-1,j) ;

  int E = ( i<(nx_d-1)  )                  ? I2D(nx_d,i+1,j) : I2D(nx_d,i-1,j) ;
  int Ex = ( i<(nx_d-2) )                  ? I2D(nx_d,i+2,j) : ((i==(nx_d-2)) ? I2D(nx_d,i,j) : I2D(nx_d,i-1,j));
  int Ey = ( (i<(nx_d-1)) && (j<(ny_d-1)) )  ? I2D(nx_d,i+1,j+1) : I2D(nx_d,i-1,j-1) ;
  int Exy = ( (i<(nx_d-2)) && (j<(ny_d-1)) ) ? I2D(nx_d,i+2,j+1) : ( (i==(nx_d-2)) ? I2D(nx_d,i,j-1) : I2D(nx_d,i-1,j-1)) ;

  gx1 = (g_p[E] - g_p[W])*invdx_d;
  gy1 = (g_p[N] - g_p[S])*invdy_d;

  gx2 = (g_p[Ex] - g_p[Wx])*invdx_d;
  gy2 = (g_p[Nx] - g_p[Sx])*invdy_d;

  gx3 = (g_p[Ey] - g_p[Wy])*invdx_d;
  gy3 = (g_p[Ny] - g_p[Sy])*invdy_d;

  gx4 = (g_p[Exy] - g_p[Wxy])*invdx_d;
  gy4 = (g_p[Nxy] - g_p[Sxy])*invdy_d;

  g.x = (1.0-s)*(1.0-t)*gx1 + s*(1.0-t)*gx2 + t*(1.0-s)*gx3 + s*t*gx4;
  g.y = (1.0-s)*(1.0-t)*gy1 + s*(1.0-t)*gy2 + t*(1.0-s)*gy3 + s*t*gy4;

  return g;

}


void tip_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
   stateVar gOut_d, stateVar gIn_d, stateVar velTan, REAL physicalTime,
   int tipAlgorithm, bool recordTip, bool *tip_plot, int *tip_count, vec5dyn *tip_vector) {

  CudaSafeCall(cudaMemset(tip_count,0,sizeof(int))); // Initialize number of contour points
  int *tip_pts;
  tip_pts = (int*)malloc(sizeof(int));
  CudaSafeCall(cudaMemset(tip_vector,0,(*tip_pts)*sizeof(vec5dyn)));
  free(tip_pts);

  // CudaSafeCall(cudaMemset(tip_count,0,sizeof(int))); // Initialize number of contour points
  // CudaSafeCall(cudaMemset(tip_vector,0,sizeof(vec5dyn)));
  // CudaSafeCall(cudaMemset(tip_plot,0,nx*ny*sizeof(bool))); // Reset screen contour plot

  switch ( tipAlgorithm ) {

    case 1:
    spiralTip_kernel<<<grid2D, block2D>>>(pitch,gIn_d.u,gOut_d.u,
      tip_plot,tip_count,tip_vector,physicalTime);
    CudaCheckError();
    // spiralTip_kernel<<<grid2D, block2D>>>(pitch,gIn_d.u,gIn_d.v,
    //   tip_plot,tip_count,tip_vector,physicalTime);
    // CudaCheckError();
    break;

    case 2:
    spiralTipNewton_kernel<<<grid2D, block2D>>>(pitch,gIn_d.u,gOut_d.u,
      tip_plot,tip_count,tip_vector,physicalTime);
    CudaCheckError();
    break;

    case 3:
    abouzarTip_kernel<<<grid2D, block2D>>>(pitch,gIn_d.u,gOut_d.u,
      tip_plot,tip_count,tip_vector,physicalTime);
    CudaCheckError();
    break;

    default:
      puts("No tip algorithm selected");

  }

}