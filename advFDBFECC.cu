
#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

extern __constant__ int nx_d, ny_d;
extern __constant__ REAL dt_d, hx_d, hy_d;
extern __constant__ REAL boundaryVal_d;
extern __constant__ bool solidSwitch_d, neumannBC_d;
extern __constant__ REAL tc_d;

__global__ void advFDBFECC_kernel(size_t pitch, stateVar g_out, stateVar g_in,  
	advVar adv, stateVar uf, stateVar ub, stateVar ue, bool *solid) {

	REAL cx, cy, FDx, FDy, Rx, Ry;

	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;

	if ( (i<nx_d) && (j<ny_d) ) {

	int i2d = i + nx_d*j;

	cx = -adv.x[i2d];
	cy = -adv.y[i2d];

	Rx = sign(cx)*cx*dt_d/hx_d;
	Ry = sign(cy)*cy*dt_d/hy_d;

	if ( neumannBC_d ) {

		if ( solidSwitch_d ) {

			/////////////////////////////////////////////
			/// With complex boundary
			/////////////////////////////////////////////

		    bool sc = solid[i2d];
			bool sw = solid[I2D(nx_d,i-1,j)];
		    bool se = solid[I2D(nx_d,i+1,j)];
		    bool sn = solid[I2D(nx_d,i,j-1)];
		    bool ss = solid[I2D(nx_d,i,j+1)];

		    // Forward

			int sW = sc ? ( sw ? I2D(nx_d,i-1,j) : I2D(nx_d,i+1,j) ) : I2D(nx_d,i,j);
			int sE = sc ? ( se ? I2D(nx_d,i+1,j) : I2D(nx_d,i-1,j) ) : I2D(nx_d,i,j);
			int sS = sc ? ( ss ? I2D(nx_d,i,j-1) : I2D(nx_d,i,j+1) ) : I2D(nx_d,i,j);
			int sN = sc ? ( sn ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j-1) ) : I2D(nx_d,i,j);

			// Backward

			int si2dW = sc ? ( sw ? I2D(nx_d,i,j) : I2D(nx_d,i+1,j) ) : I2D(nx_d,i,j);
			int sW2 = sc ? ( sw ? I2D(nx_d,i-1,j) : I2D(nx_d,i,j) ) : I2D(nx_d,i,j);
			int si2dE = sc ? ( se ? I2D(nx_d,i,j) : I2D(nx_d,i-1,j) ) : I2D(nx_d,i,j); 
			int sE2 = sc ? ( se ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j) ) : I2D(nx_d,i,j);
			int si2dS = sc ? ( ss ? I2D(nx_d,i,j) : I2D(nx_d,i,j+1) ) : I2D(nx_d,i,j); 
			int sS2  = sc ? ( ss ? I2D(nx_d,i,j-1) : I2D(nx_d,i,j) ) : I2D(nx_d,i,j);
			int si2dN = sc ? ( sn ? I2D(nx_d,i,j) : I2D(nx_d,i,j-1) ) : I2D(nx_d,i,j);
			int sN2 = sc ? ( sn ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j) ) : I2D(nx_d,i,j);

			FDx = cx>0.0 ? g_in.u[i2d] - g_in.u[sW] : g_in.u[i2d] - g_in.u[sE];
			FDy = cy>0.0 ? g_in.u[i2d] - g_in.u[sS] : g_in.u[i2d] - g_in.u[sN];

			uf.u[i2d] = g_in.u[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			FDx = cx>0.0 ? uf.u[si2dE] - uf.u[sE2] : uf.u[si2dW] - uf.u[sW2];
			FDy = cy>0.0 ? uf.u[si2dN] - uf.u[sN2] : uf.u[si2dS] - uf.u[sS2];

			ub.u[i2d] = uf.u[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			ue.u[i2d] = g_in.u[i2d] - 0.5*(ub.u[i2d] - g_in.u[i2d]);

			FDx = cx>0.0 ? ue.u[i2d] - ue.u[sW] : ue.u[i2d] - ue.u[sE];
			FDy = cy>0.0 ? ue.u[i2d] - ue.u[sS] : ue.u[i2d] - ue.u[sN];

			g_out.u[i2d] = sc ? ue.u[i2d] - tc_d*(Rx*FDx + Ry*FDy) : 0.0;

			//////////////////////////////// Gate

			FDx = cx>0.0 ? g_in.v[i2d] - g_in.v[sW] : g_in.v[i2d] - g_in.v[sE];
			FDy = cy>0.0 ? g_in.v[i2d] - g_in.v[sS] : g_in.v[i2d] - g_in.v[sN];

			uf.v[i2d] = g_in.v[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			FDx = cx>0.0 ? uf.v[si2dE] - uf.v[sE2] : uf.v[si2dW] - uf.v[sW2];
			FDy = cy>0.0 ? uf.v[si2dN] - uf.v[sN2] : uf.v[si2dS] - uf.v[sS2];

			ub.v[i2d] = uf.v[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			ue.v[i2d] = g_in.v[i2d] - 0.5*(ub.v[i2d] - g_in.v[i2d]);

			FDx = cx>0.0 ? ue.v[i2d] - ue.v[sW] : ue.v[i2d] - ue.v[sE];
			FDy = cy>0.0 ? ue.v[i2d] - ue.v[sS] : ue.v[i2d] - ue.v[sN];

			g_out.v[i2d] = sc ? ue.v[i2d] - tc_d*(Rx*FDx + Ry*FDy) : 0.0;

		} else {

			/////////////////////////////////////////////
			/// With square boundary
			/////////////////////////////////////////////

			// Forward

			int W = ( i>0 ) ? I2D(nx_d,i-1,j) : I2D(nx_d,i+1,j);
			int E = ( i<(nx_d-1) ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i-1,j); 
			int S = ( j>0 ) ? I2D(nx_d,i,j-1) : I2D(nx_d,i,j+1);
			int N = ( j<(ny_d-1) ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j-1);

			// Backward

			int i2dW = ( i>0 ) ? I2D(nx_d,i,j) : I2D(nx_d,i+1,j);
			int W2 = ( i>0 ) ? I2D(nx_d,i-1,j) : I2D(nx_d,i,j);
			int i2dE = ( i<(nx_d-1) ) ? I2D(nx_d,i,j) : I2D(nx_d,i-1,j);
			int E2 = ( i<(nx_d-1) ) ? I2D(nx_d,i+1,j) : I2D(nx_d,i,j);
			int i2dS = ( j>0 ) ? I2D(nx_d,i,j) : I2D(nx_d,i,j+1);
			int S2 = ( j>0 ) ? I2D(nx_d,i,j-1) : I2D(nx_d,i,j);
			int i2dN = ( j<(ny_d-1) ) ? I2D(nx_d,i,j) : I2D(nx_d,i,j-1);
			int N2 = ( j<(ny_d-1) ) ? I2D(nx_d,i,j+1) : I2D(nx_d,i,j);

			FDx = cx>0.0 ? g_in.u[i2d] - g_in.u[W] : g_in.u[i2d] - g_in.u[E];
			FDy = cy>0.0 ? g_in.u[i2d] - g_in.u[S] : g_in.u[i2d] - g_in.u[N];

			uf.u[i2d] = g_in.u[i2d] - tc_d*(Rx*FDx + Ry*FDy);


			FDx = cx>0.0 ? uf.u[i2dE] - uf.u[E2] : uf.u[i2dW] - uf.u[W2];
			FDy = cy>0.0 ? uf.u[i2dN] - uf.u[N2] : uf.u[i2dS] - uf.u[S2];

			ub.u[i2d] = uf.u[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			ue.u[i2d] = g_in.u[i2d] - 0.5*(ub.u[i2d] - g_in.u[i2d]);

			FDx = cx>0.0 ? ue.u[i2d] - ue.u[W] : ue.u[i2d] - ue.u[E];
			FDy = cy>0.0 ? ue.u[i2d] - ue.u[S] : ue.u[i2d] - ue.u[N];

			g_out.u[i2d] = ue.u[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			//////////////////////////////// Gate

			FDx = cx>0.0 ? g_in.v[i2d] - g_in.v[W] : g_in.v[i2d] - g_in.v[E];
			FDy = cy>0.0 ? g_in.v[i2d] - g_in.v[S] : g_in.v[i2d] - g_in.v[N];

			uf.v[i2d] = g_in.v[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			FDx = cx>0.0 ? uf.v[i2dE] - uf.v[E2] : uf.v[i2dW] - uf.v[W2];
			FDy = cy>0.0 ? uf.v[i2dN] - uf.v[N2] : uf.v[i2dS] - uf.v[S2];

			ub.v[i2d] = uf.v[i2d] - tc_d*(Rx*FDx + Ry*FDy);

			ue.v[i2d] = g_in.v[i2d] - 0.5*(ub.v[i2d] - g_in.v[i2d]);

			FDx = cx>0.0 ? ue.v[i2d] - ue.v[W] : ue.v[i2d] - ue.v[E];
			FDy = cy>0.0 ? ue.v[i2d] - ue.v[S] : ue.v[i2d] - ue.v[N];

			g_out.v[i2d] = ue.v[i2d] - tc_d*(Rx*FDx + Ry*FDy);

		}

	} else {

	    if ( solidSwitch_d ) {

	    	bool sc = solid[i2d];
			bool sw = solid[I2D(nx_d,i-1,j)];
		    bool se = solid[I2D(nx_d,i+1,j)];
		    bool sn = solid[I2D(nx_d,i,j-1)];
		    bool ss = solid[I2D(nx_d,i,j+1)];

		    // Forward

			REAL u = sc ? g_in.u[i2d] : 0.0;
			REAL W = sc && sw ? g_in.u[I2D(nx_d,i-1,j)] : ( sc && !sw ? boundaryVal_d : 0.0 );
			REAL E = sc && se ? g_in.u[I2D(nx_d,i+1,j)] : ( sc && !se ? boundaryVal_d : 0.0 ); 
			REAL S = sc && ss ? g_in.u[I2D(nx_d,i,j-1)] : ( sc && !ss ? boundaryVal_d : 0.0 );
			REAL N = sc && sn ? g_in.u[I2D(nx_d,i,j+1)] : ( sc && !sn ? boundaryVal_d : 0.0 );

			FDx = cx>0.0 ? u - W : u - E;
			FDy = cy>0.0 ? u - S : u - N;

			uf.u[i2d] = u - tc_d*(Rx*FDx + Ry*FDy);

			// Backward

			REAL uuf = sc ? uf.u[i2d] : 0.0;
			W = sc && sw ? uf.u[I2D(nx_d,i-1,j)] : ( sc && !sw ? uf.u[i2d] : 0.0 );
			E = sc && se ? uf.u[I2D(nx_d,i+1,j)] : ( sc && !se ? uf.u[i2d] : 0.0 ); 
			S = sc && ss ? uf.u[I2D(nx_d,i,j-1)] : ( sc && !ss ? uf.u[i2d] : 0.0 );
			N = sc && sn ? uf.u[I2D(nx_d,i,j+1)] : ( sc && !sn ? uf.u[i2d] : 0.0 );

			FDx = cx>0.0 ? uuf - E : uuf - W;
			FDy = cy>0.0 ? uuf - N : uuf - S;

			ub.u[i2d] = uuf - tc_d*(Rx*FDx + Ry*FDy);

			ue.u[i2d] = u - 0.5*(ub.u[i2d] - u);

			// Forward

			REAL uue = sc ? ue.u[i2d] : 0.0;
			W = sc && sw ? ue.u[I2D(nx_d,i-1,j)] : ( sc && !sw ? boundaryVal_d : 0.0 );
			E = sc && se ? ue.u[I2D(nx_d,i+1,j)] : ( sc && !se ? boundaryVal_d : 0.0 ); 
			S = sc && ss ? ue.u[I2D(nx_d,i,j-1)] : ( sc && !ss ? boundaryVal_d : 0.0 );
			N = sc && sn ? ue.u[I2D(nx_d,i,j+1)] : ( sc && !sn ? boundaryVal_d : 0.0 );

			FDx = cx>0.0 ? uue - W : uue - E;
			FDy = cy>0.0 ? uue - S : uue - N;

			g_out.u[i2d] = sc ? uue - tc_d*(Rx*FDx + Ry*FDy) : 0.0;

			//////////////////////////////// Gate

			// Forward

			REAL v = sc ? g_in.v[i2d] : 0.0;
			W = sc && sw ? g_in.v[I2D(nx_d,i-1,j)] : ( sc && !sw ? boundaryVal_d : 0.0 );
			E = sc && se ? g_in.v[I2D(nx_d,i+1,j)] : ( sc && !se ? boundaryVal_d : 0.0 ); 
			S = sc && ss ? g_in.v[I2D(nx_d,i,j-1)] : ( sc && !ss ? boundaryVal_d : 0.0 );
			N = sc && sn ? g_in.v[I2D(nx_d,i,j+1)] : ( sc && !sn ? boundaryVal_d : 0.0 );

			FDx = cx>0.0 ? v - W : v - E;
			FDy = cy>0.0 ? v - S : v - N;

			uf.v[i2d] = v - tc_d*(Rx*FDx + Ry*FDy);

			// Backward

			REAL vvf = sc ? uf.v[i2d] : 0.0;
			W = sc && sw ? uf.v[I2D(nx_d,i-1,j)] : ( sc && !sw ? uf.v[i2d] : 0.0 );
			E = sc && se ? uf.v[I2D(nx_d,i+1,j)] : ( sc && !se ? uf.v[i2d] : 0.0 ); 
			S = sc && ss ? uf.v[I2D(nx_d,i,j-1)] : ( sc && !ss ? uf.v[i2d] : 0.0 );
			N = sc && sn ? uf.v[I2D(nx_d,i,j+1)] : ( sc && !sn ? uf.v[i2d] : 0.0 );

			FDx = cx>0.0 ? vvf - E : vvf - W;
			FDy = cy>0.0 ? vvf - N : vvf - S;

			ub.v[i2d] = vvf - tc_d*(Rx*FDx + Ry*FDy);

			ue.v[i2d] = v - 0.5*(ub.v[i2d] - v);

			// Forward

			REAL vve = sc ? ue.v[i2d] : 0.0;
			W = sc && sw ? ue.v[I2D(nx_d,i-1,j)] : ( sc && !sw ? boundaryVal_d : 0.0 );
			E = sc && se ? ue.v[I2D(nx_d,i+1,j)] : ( sc && !se ? boundaryVal_d : 0.0 ); 
			S = sc && ss ? ue.v[I2D(nx_d,i,j-1)] : ( sc && !ss ? boundaryVal_d : 0.0 );
			N = sc && sn ? ue.v[I2D(nx_d,i,j+1)] : ( sc && !sn ? boundaryVal_d : 0.0 );

			FDx = cx>0.0 ? vve - W : vve - E;
			FDy = cy>0.0 ? vve - S : vve - N;

			g_out.v[i2d] = sc ? vve - tc_d*(Rx*FDx + Ry*FDy) : 0.0;

		} else {

			// Forward

			REAL u = g_in.u[i2d];
			REAL W = i>0 ? g_in.u[I2D(nx_d,i-1,j)] : boundaryVal_d;
			REAL E = i<(nx_d-1) ? g_in.u[I2D(nx_d,i+1,j)] : boundaryVal_d; 
			REAL S = j>0 ? g_in.u[I2D(nx_d,i,j-1)] : boundaryVal_d;
			REAL N = j<(ny_d-1) ? g_in.u[I2D(nx_d,i,j+1)] : boundaryVal_d; 

			FDx = cx>0.0 ? u - W : u - E;
			FDy = cy>0.0 ? u - S : u - N;

			uf.u[i2d] = u - tc_d*(Rx*FDx + Ry*FDy);

			// Backward

			REAL uuf = uf.u[i2d];
			W = i>0 ? uf.u[I2D(nx_d,i-1,j)] : uf.u[i2d];
			E = i<(nx_d-1) ? uf.u[I2D(nx_d,i+1,j)] : uf.u[i2d]; 
			S = j>0 ? uf.u[I2D(nx_d,i,j-1)] : uf.u[i2d];
			N = j<(ny_d-1) ? uf.u[I2D(nx_d,i,j+1)] : uf.u[i2d];

			FDx = cx>0.0 ? uuf - E : uuf - W;
			FDy = cy>0.0 ? uuf - N : uuf - S;

			ub.u[i2d] = uuf - tc_d*(Rx*FDx - Ry*FDy);

			ue.u[i2d] = u - 0.5*(ub.u[i2d] - u);

			// Forward

			REAL uue = ue.u[i2d];
			W = i>0 ? ue.u[I2D(nx_d,i-1,j)] : boundaryVal_d;
			E = i<(nx_d-1) ? ue.u[I2D(nx_d,i+1,j)] : boundaryVal_d; 
			S = j>0 ? ue.u[I2D(nx_d,i,j-1)] : boundaryVal_d;
			N = j<(ny_d-1) ? ue.u[I2D(nx_d,i,j+1)] : boundaryVal_d; 

			FDx = cx>0.0 ? uue - W : uue - E;
			FDy = cy>0.0 ? uue - S : uue - N;

			g_out.u[i2d] = uue - tc_d*(Rx*FDx + Ry*FDy);

			//////////////////////////////// Gate

			// Forward

			REAL v = g_in.v[i2d];
			W = i>0 ? g_in.v[I2D(nx_d,i-1,j)] : boundaryVal_d;
			E = i<(nx_d-1) ? g_in.v[I2D(nx_d,i+1,j)] : boundaryVal_d; 
			S = j>0 ? g_in.v[I2D(nx_d,i,j-1)] : boundaryVal_d;
			N = j<(ny_d-1) ? g_in.v[I2D(nx_d,i,j+1)] : boundaryVal_d; 

			FDx = cx>0.0 ? v - W : v - E;
			FDy = cy>0.0 ? v - S : v - N;

			uf.v[i2d] = v - tc_d*(Rx*FDx + Ry*FDy);

			// Backward

			REAL vvf = uf.v[i2d];
			W = i>0 ? uf.v[I2D(nx_d,i-1,j)] : uf.v[i2d];
			E = i<(nx_d-1) ? uf.v[I2D(nx_d,i+1,j)] : uf.v[i2d]; 
			S = j>0 ? uf.v[I2D(nx_d,i,j-1)] : uf.v[i2d];
			N = j<(ny_d-1) ? uf.v[I2D(nx_d,i,j+1)] : uf.v[i2d];

			FDx = cx>0.0 ? vvf - E : vvf - W;
			FDy = cy>0.0 ? vvf - N : vvf - S;

			ub.v[i2d] = vvf - tc_d*(Rx*FDx + Ry*FDy);

			ue.v[i2d] = v - 0.5*(ub.v[i2d] - v);

			// Forward

			REAL vve = ue.v[i2d];
			W = i>0 ? ue.v[I2D(nx_d,i-1,j)] : boundaryVal_d;
			E = i<(nx_d-1) ? ue.v[I2D(nx_d,i+1,j)] : boundaryVal_d;
			S = j>0 ? ue.v[I2D(nx_d,i,j-1)] : boundaryVal_d;
			N = j<(ny_d-1) ? ue.v[I2D(nx_d,i,j+1)] : boundaryVal_d; 

			FDx = cx>0.0 ? vve - W : vve - E;
			FDy = cy>0.0 ? vve - S : vve - N;

			g_out.v[i2d] = vve - tc_d*(Rx*FDx + Ry*FDy);

		}

	}

}

}

void advFDBFECC_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
	stateVar gOut, stateVar gIn,
	advVar adv, stateVar uf, stateVar ub, stateVar ue, bool *solid) {

	advFDBFECC_kernel<<<grid2D,block2D>>>(pitch,gOut,gIn,adv,uf,ub,ue,solid);
	CudaCheckError();

}