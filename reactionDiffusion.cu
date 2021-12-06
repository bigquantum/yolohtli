
#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL dt_d, rx_d, ry_d, qx4_d, qy4_d, fx4_d, fy4_d;
extern __constant__ REAL rxy_d, rbx_d, rby_d, rscale_d;
extern __constant__ REAL tc_d, alpha_d, beta_d, delta_d, eps_d, mu_d, gamma_d, theta_d;
extern __constant__ REAL boundaryVal_d;
extern __constant__ bool solidSwitch_d, neumannBC_d, gateDiff_d, anisotropy_d;
extern __constant__ int lap4_d, timeIntOrder_d;
extern __constant__ REAL conTh1_d, conTh2_d, conTh3_d;

/*========================================================================
 * Main Entry of the Kernel
 *========================================================================
*/

__global__ void reactionDiffusion_kernel(size_t pitch,
  stateVar g_out, stateVar g_in, stateVar J,
  stateVar velTan, bool reduceSym, bool *solid, bool stimLock, REAL *stim,
  bool stimLockMouse, int2 point) {

  /*------------------------------------------------------------------------
  * getting i and j global indices
  *-------------------------------------------------------------------------
  */

  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

  /*------------------------------------------------------------------------
  * converting global index into matrix indices assuming
  * the column major structure of the matlab matrices
  *-------------------------------------------------------------------------
  */

  const int i2d = i + j*nx_d;

  /*------------------------------------------------------------------------
  * Circular mouse stimulus
  *------------------------------------------------------------------------
  */

  bool scs;

  int ic = i-nx_d/2;
  int jc = j-ny_d/2;
  int cx = __float2int_rn(point.x-nx_d/2);
  int cy = __float2int_rn(point.y-ny_d/2);

  scs = ( ( (ic-cx)*(ic-cx) + (jc-cy)*(jc-cy) ) < 400 ) && stimLockMouse ? true : false;


  /*------------------------------------------------------------------------
  * RK4 arrays
  *------------------------------------------------------------------------
  */

  REAL ki[4], sumrk4[4];

  switch (timeIntOrder_d) {

    case 1: // Euler
      ki[0] = 0.0; ki[1] = 0.0; ki[2] = 0.0; ki[3] = 0.0;
      sumrk4[0] = 1.0; sumrk4[1] = 0.0; sumrk4[2] = 0.0; sumrk4[3] = 0.0;
    break;
    case 2: // RK2
      ki[0] = 0.0; ki[1] = 0.5; ki[2] = 0.0; ki[3] = 0.0;
      sumrk4[0] = 0.0; sumrk4[1] = 1.0;
      sumrk4[2] = 0.0; sumrk4[3] = 0.0;
    break;
    case 4: // RK4
      ki[0] = 0.0; ki[1] = 0.5; ki[2] = 0.5; ki[3] = 1.0;
      sumrk4[0] = 0.166666666666667; sumrk4[1] = 0.333333333333333;
      sumrk4[2] = 0.333333333333333; sumrk4[3] = 0.166666666666667;
    break;
    default:
      if ( (threadIdx.x==0) && (threadIdx.y==0) ) {
        printf("No time integration method selected\n");
      }
    break;

  }

  /*------------------------------------------------------------------------
  * Setting local variables
  *-------------------------------------------------------------------------
  */

  REAL du2dt = 0.0;
  REAL dv2dt = 0.0;
  REAL rhs_u = 0.0;
  REAL rhs_v = 0.0;
  REAL u, v;
  REAL I_sum, I_v;

  REAL u0 = g_in.u[i2d] ;
  REAL v0 = g_in.v[i2d] ;

  /*------------------------------------------------------------------------
  * Euler/RK4 loop starts
  *-------------------------------------------------------------------------
  */

  for (int rk4idx=0;rk4idx<timeIntOrder_d;rk4idx++) {

  g_in.u[i2d] = u0 + ( ki[rk4idx] * du2dt );
  g_in.v[i2d] = v0 + ( ki[rk4idx] * dv2dt );

  u = g_in.u[i2d];
  v = g_in.v[i2d];

  du2dt = 0.0;
  dv2dt = 0.0;

  /*------------------------------------------------------------------------
  * I_sum (voltage)
  *-------------------------------------------------------------------------
  */

  I_sum = -( mu_d*u*(1.0-u)*(u-alpha_d) - u*v )
    // - ( stimLock ? stim[i2d] : 0.0 );
    // -( stimLock ? ( (u<conTh1_d) && (abs(v-conTh2_d)<conTh3_d) ? stim[i2d] : 0.0 ) : 0.0 );
    - ( scs ? 24.7 : 0.0 ); // Mouse stimulus

  /*------------------------------------------------------------------------
  * Calculating the reaction for the gates
  *------------------------------------------------------------------------
  */

  I_v = -( eps_d*(delta_d*(u-gamma_d)*(beta_d-u) - v - theta_d) );

  /////////////////////////////
  // Solve homogeneous tissue
  /////////////////////////////

  if ( neumannBC_d ) {

    int S = I2D(nx_d,i,coord_j(j-1));
    int N = I2D(nx_d,i,coord_j(j+1));
    int W = I2D(nx_d,coord_i(i-1),j);
    int E = I2D(nx_d,coord_i(i+1),j);

    if ( solidSwitch_d ) {

      bool sc = solid[i2d];
      bool sw = solid[W];
      bool se = solid[E];
      bool sn = solid[N];
      bool ss = solid[S];
        
      float3 coeffx =
      make_float3( (sw && se) && (sw && sc) ? 1.0 : ( (sw && sc) ? 2.0 : 0.0) ,
                    sc ? ( (sw || se) ? 2.0 : 0.0 ) : 0.0 ,
                    (sw && se) && (sc && se) ? 1.0 : ( (sc && se) ? 2.0 : 0.0 ));
      float3 coeffy =
      make_float3( (sn && ss) && (sn && sc) ? 1.0 : ( (sn && sc) ? 2.0 : 0.0 ) ,
                    sc ? ( (sn || ss) ? 2.0 : 0.0 ) : 0.0 ,
                    (sn && ss) && (sc && ss) ? 1.0 : ( (sc && ss) ? 2.0 : 0.0 ));

      du2dt = (
           ( coeffx.x*g_in.u[W] - coeffx.y*u + coeffx.z*g_in.u[E] )*rx_d
      +    ( coeffy.x*g_in.u[N] - coeffy.y*u + coeffy.z*g_in.u[S] )*ry_d );

      if ( gateDiff_d ) {

        // Gate 1
        dv2dt = (
             ( coeffx.x*g_in.v[W] - coeffx.y*v + coeffx.z*g_in.v[E] )*rx_d*rscale_d
        +    ( coeffy.x*g_in.v[N] - coeffy.y*v + coeffy.z*g_in.v[S] )*ry_d*rscale_d );

      }

      // Anisotropic mode pending FIXME

    } else { // Square boundary

      du2dt = (
         ( g_in.u[W] - 2.0*u + g_in.u[E] )*rx_d
      +  ( g_in.u[N] - 2.0*u + g_in.u[S] )*ry_d );

      if ( gateDiff_d ) {

        // Gate 1
        dv2dt = (
           ( g_in.v[W] - 2.0*v + g_in.v[E] )*rx_d*rscale_d
        +  ( g_in.v[N] - 2.0*v + g_in.v[S] )*ry_d*rscale_d );

      }

      if ( lap4_d ) {

        int SWxy = (i>0   && j>0)?  I2D(nx_d,i-1,j-1) :
                  ((i==0  && j>0)?  I2D(nx_d,i+1,j-1) :
                  ((i>0   && j==0)? I2D(nx_d,i-1,j+1) : I2D(nx_d,i+1,j+1) ) ) ;

        int SExy = (i<(nx_d-1)  && j>0)?  I2D(nx_d,i+1,j-1) :
                  ((i==(nx_d-1) && j>0)?  I2D(nx_d,i-1,j-1) :
                  ((i<(nx_d-1)  && j==0)? I2D(nx_d,i+1,j+1) : I2D(nx_d,i-1,j+1) ) ) ;

        int NWxy = (i>0   && j<(ny_d-1))?   I2D(nx_d,i-1,j+1) :
                  ((i==0  && j<(ny_d-1))?   I2D(nx_d,i+1,j+1) :
                  ((i>0   && j==(ny_d-1))?  I2D(nx_d,i-1,j-1) : I2D(nx_d,i+1,j-1) ) ) ;

        int NExy = (i<(nx_d-1)  && j<(ny_d-1))?   I2D(nx_d,i+1,j+1) :
                  ((i==(nx_d-1) && j<(ny_d-1))?   I2D(nx_d,i-1,j+1) :
                  ((i<(nx_d-1)  && j==(ny_d-1))?  I2D(nx_d,i+1,j-1) : I2D(nx_d,i-1,j-1) ) ) ;

        J.u[i2d] = I_sum;

        du2dt += -2.0*( qx4_d+qy4_d )*(
              +  ( g_in.u[W] - u + g_in.u[E] )
              +  ( g_in.u[N] - u + g_in.u[S] ) );

        du2dt += ( qx4_d+qy4_d )*(g_in.u[SWxy]+g_in.u[SExy]+g_in.u[NWxy]+g_in.u[NExy]);

        du2dt -= (
                 ( J.u[W] - 2.0*I_sum + J.u[E] )*fx4_d
              +  ( J.u[N] - 2.0*I_sum + J.u[S] )*fy4_d );

        if ( gateDiff_d ) {

          J.v[i2d] = I_v;

          dv2dt += -rscale_d*2.0*( qx4_d+qy4_d )*(
                +  ( g_in.v[W] - v + g_in.v[E] )
                +  ( g_in.v[N] - v + g_in.v[S] ) );

          dv2dt += rscale_d*( qx4_d+qy4_d )*(g_in.v[SWxy]+g_in.v[SExy]+g_in.v[NWxy]+g_in.v[NExy]);

          dv2dt -= (
                   ( J.v[W] - 2.0*I_v + J.v[E] )*fx4_d
                +  ( J.v[N] - 2.0*I_v + J.v[S] )*fy4_d );

        }

      }

      if ( anisotropy_d ) {

        int SWxy = (i>0  && j>0) ? I2D(nx_d,i-1,j-1) :
                  ((i==0 && j>0) ? I2D(nx_d,i+1,j-1) :
                  ((i>0  && j==0)? I2D(nx_d,i-1,j+1) : I2D(nx_d,i+1,j+1) ) ) ;

        int SExy = (i<(nx_d-1)  && j>0) ? I2D(nx_d,i+1,j-1) :
                  ((i==(nx_d-1) && j>0) ? I2D(nx_d,i-1,j-1) :
                  ((i<(nx_d-1)  && j==0)? I2D(nx_d,i+1,j+1) : I2D(nx_d,i-1,j+1) ) ) ;

        int NWxy = (i>0  && j<(ny_d-1)) ? I2D(nx_d,i-1,j+1) :
                  ((i==0 && j<(ny_d-1)) ? I2D(nx_d,i+1,j+1) :
                  ((i>0  && j==(ny_d-1))? I2D(nx_d,i-1,j-1) : I2D(nx_d,i+1,j-1) ) ) ;

        int NExy = (i<(nx_d-1)  && j<(ny_d-1)) ? I2D(nx_d,i+1,j+1) :
                  ((i==(nx_d-1) && j<(ny_d-1)) ? I2D(nx_d,i-1,j+1) :
                  ((i<(nx_d-1)  && j==(ny_d-1))? I2D(nx_d,i+1,j-1) : I2D(nx_d,i-1,j-1) ) ) ;

        REAL b_S = (j > 0 )? 0.0:
                  ((j==0 && (i==0 || i==(nx_d-1)))? 0.0:
                  rby_d*(g_in.u[I2D(nx_d,i+1,j)] - g_in.u[I2D(nx_d,i-1,j)])) ;

        REAL b_N = (j < (ny_d-1))? 0.0:
                  ((j==(ny_d-1) && (i==0 || i==(nx_d-1)))? 0.0:
                  -rby_d*(g_in.u[I2D(nx_d,i+1,j)] - g_in.u[I2D(nx_d,i-1,j)])) ;

        REAL b_W = (i > 0 )? 0.0:
                  ((i==0 && (j==0 || j==(ny_d-1)))? 0.0:
                  rbx_d*(g_in.u[I2D(nx_d,i,j+1)] - g_in.u[I2D(nx_d,i,j-1)])) ;

        REAL b_E = (i < (nx_d-1))? 0.0:
                  ((i==(nx_d-1) && (j==0 || j==(ny_d-1)))? 0.0:
                  -rbx_d*(g_in.u[I2D(nx_d,i,j+1)] - g_in.u[I2D(nx_d,i,j-1)])) ;

        du2dt += (
                 ( b_S + b_N )*ry_d
             +   ( b_W + b_E )*rx_d  );

        // Correcion to SW SE NW NE boundary conditions
        REAL b_SW = (i>0  && j>0)?  0.0 :
                   ((i==0 && j>1)?  rbx_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i,j-2)]) :
                   ((i>1  && j==0)? rby_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i-2,j)]) : 0.0)) ;

        REAL b_SE = (i<(nx_d-1)  && j>0)?  0.0 :
                   ((i==(nx_d-1) && j>1)? -rbx_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i,j-2)]) :
                   ((i<(nx_d-2)  && j==0)? rby_d*(g_in.u[I2D(nx_d,i+2,j)] - g_in.u[i2d]) : 0.0)) ;

        REAL b_NW = (i>0  && j<(ny_d-1)) ?  0.0 :
                   ((i==0 && j<(ny_d-2)) ?  rbx_d*(g_in.u[I2D(nx_d,i,j+2)] - g_in.u[i2d]) :
                   ((i>1  && j==(ny_d-1))? -rby_d*(g_in.u[i2d] - g_in.u[I2D(nx_d,i-2,j)]) : 0.0)) ;

        REAL b_NE = (i<(nx_d-1)  && j<(ny_d-1)) ? 0.0 :
                   ((i==(nx_d-1) && j<(ny_d-2)) ? -rbx_d*(g_in.u[I2D(nx_d,i,j+2)] - g_in.u[i2d]) :
                   ((i<(nx_d-2)  && j==(ny_d-1))? -rby_d*(g_in.u[I2D(nx_d,i+2,j)] - g_in.u[i2d]) : 0.0)) ;

        du2dt += ( rxy_d * ( (g_in.u[SWxy] + b_SW) +
                             (g_in.u[NExy] + b_NE) -
                             (g_in.u[SExy] + b_SE) -
                             (g_in.u[NWxy] + b_NW) ) );

        if ( gateDiff_d ) {

          // Gate 1
          REAL b_S = (j > 0 )? 0.0:
                    ((j==0 && (i==0 || i==(nx_d-1)))? 0.0:
                    rby_d*(g_in.v[I2D(nx_d,i+1,j)] - g_in.v[I2D(nx_d,i-1,j)])) ;

          REAL b_N = (j < (ny_d-1))? 0.0:
                    ((j==(ny_d-1) && (i==0 || i==(nx_d-1)))? 0.0:
                    -rby_d*(g_in.v[I2D(nx_d,i+1,j)] - g_in.v[I2D(nx_d,i-1,j)])) ;

          REAL b_W = (i > 0 )? 0.0:
                    ((i==0 && (j==0 || j==(ny_d-1)))? 0.0:
                    rbx_d*(g_in.v[I2D(nx_d,i,j+1)] - g_in.v[I2D(nx_d,i,j-1)])) ;

          REAL b_E = (i < (nx_d-1))? 0.0:
                    ((i==(nx_d-1) && (j==0 || j==(ny_d-1)))? 0.0:
                    -rbx_d*(g_in.v[I2D(nx_d,i,j+1)] - g_in.v[I2D(nx_d,i,j-1)])) ;

          dv2dt += (
                   ( b_S + b_N )*ry_d
               +   ( b_W + b_E )*rx_d  );

          // Correcion to SW SE NW NE boundary conditions
          REAL b_SW = (i>0  && j>0)?  0.0 :
                     ((i==0 && j>1)?  rbx_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i,j-2)]) :
                     ((i>1  && j==0)? rby_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i-2,j)]) : 0.0)) ;

          REAL b_SE = (i<(nx_d-1)  && j>0)?  0.0 :
                     ((i==(nx_d-1) && j>1)? -rbx_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i,j-2)]) :
                     ((i<(nx_d-2)  && j==0)? rby_d*(g_in.v[I2D(nx_d,i+2,j)] - g_in.v[i2d]) : 0.0)) ;

          REAL b_NW = (i>0  && j<(ny_d-1)) ?  0.0 :
                     ((i==0 && j<(ny_d-2)) ?  rbx_d*(g_in.v[I2D(nx_d,i,j+2)] - g_in.v[i2d]) :
                     ((i>1  && j==(ny_d-1))? -rby_d*(g_in.v[i2d] - g_in.v[I2D(nx_d,i-2,j)]) : 0.0)) ;

          REAL b_NE = (i<(nx_d-1)  && j<(ny_d-1)) ? 0.0 :
                     ((i==(nx_d-1) && j<(ny_d-2)) ? -rbx_d*(g_in.v[I2D(nx_d,i,j+2)] - g_in.v[i2d]) :
                     ((i<(nx_d-2)  && j==(ny_d-1))? -rby_d*(g_in.v[I2D(nx_d,i+2,j)] - g_in.v[i2d]) : 0.0)) ;

          dv2dt += ( rxy_d * ( (g_in.v[SWxy] + b_SW) +
                               (g_in.v[NExy] + b_NE) -
                               (g_in.v[SExy] + b_SE) -
                               (g_in.v[NWxy] + b_NW) )*rscale_d );

        }

      }

    }

  } else { // Dirichlet bc

    int S = I2D(nx_d,i,j-1);
    int N = I2D(nx_d,i,j+1);
    int W = I2D(nx_d,i-1,j);
    int E = I2D(nx_d,i+1,j);

    if ( solidSwitch_d ) {

      bool sc = solid[i2d];
      bool sw = solid[W];
      bool se = solid[E];
      bool sn = solid[N];
      bool ss = solid[S];

      REAL uS = sc && ss ? g_in.u[S] : boundaryVal_d;
      REAL uN = sc && sn ? g_in.u[N] : boundaryVal_d;
      REAL uW = sc && sw ? g_in.u[W] : boundaryVal_d;
      REAL uE = sc && se ? g_in.u[E] : boundaryVal_d;

      du2dt = (
          ( uW - 2.0*u + uE )*rx_d
      +   ( uN - 2.0*u + uS )*ry_d );

      if ( gateDiff_d ) {

        REAL vS = sc && ss ? g_in.v[S] : boundaryVal_d;
        REAL vN = sc && sn ? g_in.v[N] : boundaryVal_d;
        REAL vW = sc && sw ? g_in.v[W] : boundaryVal_d;
        REAL vE = sc && se ? g_in.v[E] : boundaryVal_d;

        dv2dt = (
             ( vW - 2.0*v + vE )*rx_d*rscale_d
        +    ( vN - 2.0*v + vS )*ry_d*rscale_d );

      }

      if ( anisotropy_d ) {

        int SW = I2D(nx_d,i-1,j-1);
        int SE = I2D(nx_d,i+1,j-1);
        int NW = I2D(nx_d,i-1,j+1);
        int NE = I2D(nx_d,i+1,j+1);

        bool ssw = solid[SW];
        bool sse = solid[SE];
        bool snw = solid[NW];
        bool sne = solid[NE];

        REAL uSWxy = sc && ssw ? g_in.u[SW] : boundaryVal_d ;
        REAL uSExy = sc && sse ? g_in.u[SE] : boundaryVal_d ;
        REAL uNWxy = sc && snw ? g_in.u[NW] : boundaryVal_d ;
        REAL uNExy = sc && sne ? g_in.u[NE] : boundaryVal_d ;

        du2dt += ( rxy_d * (  uSWxy +
                              uNExy -
                              uSExy -
                              uNWxy ) );

        if ( gateDiff_d ) {

          REAL vSWxy = sc && ssw ? g_in.v[SW] : boundaryVal_d ;
          REAL vSExy = sc && sse ? g_in.v[SE] : boundaryVal_d ;
          REAL vNWxy = sc && snw ? g_in.v[NW] : boundaryVal_d ;
          REAL vNExy = sc && sne ? g_in.v[NE] : boundaryVal_d ;

          dv2dt += ( rxy_d * (  vSWxy +
                                vNExy -
                                vSExy -
                                vNWxy )*rscale_d );

        }

      }

    } else { // Square boundary

      REAL uS = j>0 ? g_in.u[S] : boundaryVal_d;
      REAL uN = j<(ny_d-1) ? g_in.u[N] : boundaryVal_d;
      REAL uW = i>0 ? g_in.u[W] : boundaryVal_d;
      REAL uE = i<(nx_d-1) ? g_in.u[E] : boundaryVal_d;

      du2dt = (
           ( uW - 2.0*u + uE )*rx_d
      +    ( uN - 2.0*u + uS )*ry_d );

      if ( gateDiff_d ) {

        // Gate 1
        REAL vS = j>0 ? g_in.v[S] : boundaryVal_d;
        REAL vN = j<(ny_d-1) ? g_in.v[N] : boundaryVal_d;
        REAL vW = i>0 ? g_in.v[W] : boundaryVal_d;
        REAL vE = i<(nx_d-1) ? g_in.v[E] : boundaryVal_d;

        dv2dt = (
             ( vW - 2.0*v + vE )*rx_d*rscale_d
        +    ( vN - 2.0*v + vS )*ry_d*rscale_d );

      }

      if ( anisotropy_d ) {

        REAL uSWxy = (i>0) && (j>0) ? g_in.u[I2D(nx_d,i-1,j-1)] : boundaryVal_d ;
        REAL uSExy = (i<(nx_d-1)) && (j>0) ? g_in.u[I2D(nx_d,i+1,j-1)] : boundaryVal_d ;
        REAL uNWxy = (i>0) && (j<(ny_d-1)) ? g_in.u[I2D(nx_d,i-1,j+1)] : boundaryVal_d ;
        REAL uNExy = (i<(nx_d-1)) && (j<(ny_d-1)) ? g_in.u[I2D(nx_d,i+1,j+1)] : boundaryVal_d ;

        du2dt += ( rxy_d * (  uSWxy +
                              uNExy -
                              uSExy -
                              uNWxy ) );

        if ( gateDiff_d ) {

          REAL vSWxy = (i>0) && (j>0) ? g_in.v[I2D(nx_d,i-1,j-1)] : boundaryVal_d ;
          REAL vSExy = (i<(nx_d-1)) && (j>0) ? g_in.v[I2D(nx_d,i+1,j-1)] : boundaryVal_d ;
          REAL vNWxy = (i>0) && (j<(ny_d-1)) ? g_in.v[I2D(nx_d,i-1,j+1)] : boundaryVal_d ;
          REAL vNExy = (i<(nx_d-1)) && (j<(ny_d-1)) ? g_in.v[I2D(nx_d,i+1,j+1)] : boundaryVal_d ;

          dv2dt += ( rxy_d * (  vSWxy +
                                vNExy -
                                vSExy -
                                vNWxy )*rscale_d );

        }

      }

    }

  }

  /*------------------------------------------------------------------------
  * RHS
  *------------------------------------------------------------------------
  */

  // Compute the right hand side of the equations
  du2dt -= dt_d*I_sum ; // f(t, y_n + 1/2*k)
  dv2dt -= dt_d*I_v ;

  // RK4 final sum
  rhs_u += ( sumrk4[rk4idx]*du2dt );
  rhs_v += ( sumrk4[rk4idx]*dv2dt );

  } // RK4 loop ends

  // Retrieve initial condition
  g_in.u[i2d] = u0;
  g_in.v[i2d] = v0;

  // Update
  u0 += tc_d*rhs_u;
  v0 += tc_d*rhs_v;

  if ( solidSwitch_d ) {

    bool sc = solid[i2d];

    if ( gateDiff_d ) {

      g_out.u[i2d] = sc ? u0 : 0.0;
      g_out.v[i2d] = sc ? v0 : 0.0;

      /*------------------------------------------------------------------------
      * Calculate velocity tangent
      *------------------------------------------------------------------------
      */

      velTan.u[i2d] = sc ? rhs_u / dt_d : 0.0;
      velTan.v[i2d] = sc ? rhs_v / dt_d : 0.0;

    } else {

      g_out.u[i2d] = sc ? u0 : 0.0;
      g_out.v[i2d] = sc ? v0 : 0.0;

    }

  } else {

    if ( gateDiff_d ) {

      g_out.u[i2d] = u0;
      g_out.v[i2d] = v0;

      /*------------------------------------------------------------------------
      * Calculate velocity tangent
      *------------------------------------------------------------------------
      */

      velTan.u[i2d] = rhs_u / dt_d ;
      velTan.v[i2d] = rhs_v / dt_d ;

    } else {

      g_out.u[i2d] = u0;
      g_out.v[i2d] = v0;

    }

  }


}

}

void reactionDiffusion_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
   stateVar gOut_d, stateVar gIn_d, stateVar J,
   stateVar velTan, bool reduceSym, bool *solid, bool stimLock, REAL *stim, 
   bool stimLockMouse, int2 point) {

  reactionDiffusion_kernel<<<grid2D, block2D>>>(pitch,gOut_d,gIn_d,J,
    velTan,reduceSym,solid,stimLock,stim,stimLockMouse,point);
  CudaCheckError();

}


