

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "typeDefinition.cuh"
// #include "globalVariables.cuh"

// Performance libraries
#include "./common/CudaSafeCall.h"
#include "./common/profile_time.h"

// Function protoypes
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"
#include "openGLPrototypes.h"

using namespace std;

/*------------------------------------------------------------------------
* Decalre global variables
*------------------------------------------------------------------------
*/

size_t pitch;

fileVar strAdress;

paramVar param;

// Constants for device
__constant__ int nx_d, ny_d;
__constant__ REAL dt_d, rx_d, ry_d, hx_d, hy_d, Lx_d, Ly_d, qx4_d, qy4_d, fx4_d, fy4_d;
__constant__ REAL rxy_d, rbx_d, rby_d, rscale_d;
__constant__ REAL invdx_d, invdy_d;
__constant__ REAL tc_d, alpha_d, beta_d, delta_d, eps_d, mu_d, gamma_d, theta_d;
__constant__ REAL boundaryVal_d;
__constant__ bool solidSwitch_d, neumannBC_d, gateDiff_d, anisotropy_d, tipGrad_d;
__constant__ int tipOffsetX_d, tipOffsetY_d;
__constant__ float minVarColor_d, maxVarColor_d;
__constant__ float tipx0_d, tipy0_d;
__constant__ REAL Uth_d, conTh1_d, conTh2_d, conTh3_d;
__constant__ int lap4_d, timeIntOrder_d;

// Kernel setup
dim3 grid0D, block0D, grid1D, block1D, grid2D, block2D;

// Voltage and gate arrays
stateVar gate_h, gateIn_d, gateOut_d, J_d;

// Slices (templates) for the integrals
REAL3 c, c0, phi, phi0;
sliceVar slice, slice0;
stateVar velTan;
REAL *integrals;
std::vector<REAL3> clist;
std::vector<REAL3> philist;

// Advection arrays
advVar advect;
// BFECC (advection integration) arrays
stateVar uf_d, ub_d, ue_d;

// Solid holes //
bool *solid, *solid_d; // For the PDEs
REAL *coeffTrapz, *coeffTrapz_d; // For the integrals
bool *intglArea, *intglArea_d; // For the slices (derivatives)

// Arrays for the tip trajectory
int *tip_count_d;
vec5dyn *tip_vector_d; // Holds the tip trajectory
bool *tip_plot;

// Single cell recordings
std::vector<electrodeVar> electrode;
std::vector<REAL> contourLength;
REAL *point_h, *point_d;
stateVar timeSeries;

// Arrays for the contour tracking
int *contour_count_d;
float3 *contour_vector_d; // Holds the contours
bool *stimArea, *stimArea_d; // Area considered for APD(x,y)
REAL *APD1_d, *APD2_d, *sAPD_d, *dAPD_d, *back_d, *front_d; // sAPD_d: APD spatial field
bool *first_APD_d, *contour_plot; // Auxiliary array. Contour plots
REAL *stimulus, *stimulus_d; // Stimulus area
int sampleIt; // Sample period in iterations per unit time
int stimPeriodIt;
int stimDurationIt;
int stimCount = 0;
bool stimLock = false, stimLockKey = false, stimLockMouse = false;
REAL *timeSeriesContourLength;

// Figures/shapes
float2 *trapzAreaCircle, *stimAreaCircle;

// OpenGL pixel buffer object and texture //
GLuint gl_PBO, gl_Tex;
unsigned int *cmap_rgba; // rgba arrays for plotting
unsigned int *cmap_rgba_data, *plot_rgba_data;
float width, height;
int nsolid, nstep, nsteps, ncol;
GLint window1, window2;
bool timeScreen = true, phaseScreen = false, contourScreen = false;
REAL *timeSeriesCx, *timeSeriesCy, *timeSeriesCt;

// Timer for simulation
int base;
float FPS = 1.0;
int initial_time = time(NULL); // Timer for controlling the fps
int final_time, frame_count; // More fram/fps stuff
int fps;

/*------------------------------------------------------------------------
* Program starts here
*------------------------------------------------------------------------
*/

int main(int argc, char *argv[]) {

if (argc > 1) {
  printf("%d VALUES(s) has been introduced form the EXECUTABLE COMMAND line\n",argc-1);
}

param = startMenu(&strAdress,param,argc,argv);

if ( param.save ) {
  strAdress = saveFileNames(strAdress,&param);
}

/*------------------------------------------------------------------------
* Parameters form command line
*------------------------------------------------------------------------
*/

if (argc > 1) param.stimPeriod = atof(argv[1]);


// Commonly changed variables
param.dt = param.reduceSym ? 0.5*param.dt : param.dt;
param.rx = param.dt*param.Dxx/(param.hx*param.hx);
param.ry = param.dt*param.Dyy/(param.hy*param.hy);
param.rxy = 2.0*param.Dxy*param.dt/(4.0*param.hx*param.hy);
param.qx4 = param.dt*param.Dyy/(param.hy*param.hy*12.0);
param.qy4 = param.dt*param.Dxx/(param.hx*param.hx*12.0);
param.fx4 = param.dt/12.0;
param.fy4 = param.dt/12.0;
sampleIt = round(param.sample/param.dt);
stimPeriodIt = floor(param.stimPeriod/param.dt);
stimDurationIt = floor(param.stimDuration/param.dt);

// Symmetry variables (also for predictor scheme)
c.x = 0.0; c.y = 0.0; c.t = 0.0;
c0.x = 0.0; c0.y = 0.0; c0.t = 0.0;
phi.x = 0.0; phi.y = 0.0; phi.t = 0.0;
phi0.x = 0.0; phi0.y = 0.0; phi0.t = 0.0;

// Kernel setup
grid0D = dim3(1,1,1);
block0D = dim3(1,1,1);
grid1D = dim3(GRIDSIZE_1D,1,1);
block1D = dim3(BLOCKSIZE_1D,1,1);
grid2D = dim3(iDivUp(param.nx,BLOCK_DIM_X),iDivUp(param.ny,BLOCK_DIM_Y),1);
block2D = dim3(BLOCK_DIM_X, BLOCK_DIM_Y,1);

printf("\n**Parameter values**\n");
printf("dt = %f ms \n", param.dt);
printf("rx = %f \n", param.rx);
printf("ry = %f \n", param.ry);
printf("Lx = %f cm \n", param.Lx);
printf("Ly = %f cm \n", param.Ly);
printf("nx = %d \n", param.nx);
printf("ny = %d \n", param.ny);
printf("hx = %f cm \n", param.hx);
printf("hy = %f cm \n", param.hy);
printf("Dx = %f cm^2/ms \n",param.Dxx);
printf("Dy = %f cm^2/ms \n",param.Dyy);

printf("\n");
pressEnterKey();

printf("\n**Keyboard options**\n");
printf("m --> Menu\n");
printf("Esc --> Close application\n");
printf("Space bar --> Pause simulation\n");
printf("r --> Restart simulation\n");
printf("q --> Pace/stimulate\n");
printf("s --> Symmetry reduction\n");
printf("t --> Tip tracjectory recordings\n");
printf("c --> Contour recordings\n");
printf("p --> Print screenshot\n");
printf("/ --> Conduction block\n");
printf("x --> Stimulate all tissue once\n");
printf("x --> Stimulate circle tissue once\n");
printf("z --> Clear screen\n");
printf("1 --> Screen 1 (electrodes)\n");
printf("2 --> Screen 2 (SR velocities)\n");
printf("3 --> Screen 3 (Contour length)\n");

/*------------------------------------------------------------------------
* Array allocation
*------------------------------------------------------------------------
*/

param.memSize = param.nx*param.ny*sizeof(REAL);

// Array allocation
gate_h.u = (REAL*)malloc(param.memSize);
gate_h.v = (REAL*)malloc(param.memSize);

// Holds the results of the slice integrals
integrals = (REAL*)malloc(12*sizeof(REAL));

// Circular boundary
solid = new bool[param.nx*param.ny];
coeffTrapz = (REAL*)malloc(param.memSize);
intglArea = new bool[param.nx*param.ny];

// Stimulation area
stimArea = new bool[param.nx*param.ny];
stimulus = (REAL*)malloc(param.memSize);

// Electrode recordings
point_h = (REAL*)malloc(param.eSize*sizeof(REAL));
timeSeries.u = (REAL*)malloc(param.wnx*sizeof(REAL));
timeSeries.v = (REAL*)malloc(param.wnx*sizeof(REAL));

// Array for figures/shapes
trapzAreaCircle = (float2*)malloc(param.nc*sizeof(float2));
stimAreaCircle = (float2*)malloc(param.nc*sizeof(float2));

// Plotting cx, cy, ct
timeSeriesCx = (REAL*)malloc(param.wnx*sizeof(REAL));
timeSeriesCy = (REAL*)malloc(param.wnx*sizeof(REAL));
timeSeriesCt = (REAL*)malloc(param.wnx*sizeof(REAL));

// Contour length
timeSeriesContourLength = (REAL*)malloc(param.wnx*sizeof(REAL));

// Allocate device memory arrays
CudaSafeCall(cudaMalloc(&gateIn_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&gateIn_d.v,param.memSize));
CudaSafeCall(cudaMalloc(&gateOut_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&gateOut_d.v,param.memSize));
CudaSafeCall(cudaMalloc(&J_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&J_d.v,param.memSize));
CudaSafeCall(cudaMalloc(&slice.ux,param.memSize));
CudaSafeCall(cudaMalloc(&slice.uy,param.memSize));
CudaSafeCall(cudaMalloc(&slice.ut,param.memSize));
CudaSafeCall(cudaMalloc(&slice.vx,param.memSize));
CudaSafeCall(cudaMalloc(&slice.vy,param.memSize));
CudaSafeCall(cudaMalloc(&slice.vt,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.ux,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.uy,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.ut,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.vx,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.vy,param.memSize));
CudaSafeCall(cudaMalloc(&slice0.vt,param.memSize));

CudaSafeCall(cudaMalloc(&advect.x,param.memSize));
CudaSafeCall(cudaMalloc(&advect.y,param.memSize));
CudaSafeCall(cudaMalloc(&velTan.u,param.memSize));
CudaSafeCall(cudaMalloc(&velTan.v,param.memSize));

CudaSafeCall(cudaMalloc(&tip_plot, param.nx*param.ny*sizeof(bool)));

CudaSafeCall(cudaMalloc(&uf_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&ub_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&ue_d.u,param.memSize));
CudaSafeCall(cudaMalloc(&uf_d.v,param.memSize));
CudaSafeCall(cudaMalloc(&ub_d.v,param.memSize));
CudaSafeCall(cudaMalloc(&ue_d.v,param.memSize));

CudaSafeCall(cudaMalloc(&tip_count_d,sizeof(int)));
CudaSafeCall(cudaMalloc(&tip_vector_d,TIPVECSIZE*sizeof(vec5dyn)));

CudaSafeCall(cudaMalloc(&solid_d, param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMalloc(&coeffTrapz_d,param.memSize));
CudaSafeCall(cudaMalloc(&intglArea_d,param.nx*param.ny*sizeof(bool)));

CudaSafeCall(cudaMalloc(&APD1_d,param.memSize));
CudaSafeCall(cudaMalloc(&APD2_d,param.memSize));
CudaSafeCall(cudaMalloc(&sAPD_d,param.memSize));
CudaSafeCall(cudaMalloc(&dAPD_d,param.memSize));
CudaSafeCall(cudaMalloc(&back_d,param.memSize));
CudaSafeCall(cudaMalloc(&front_d,param.memSize));
CudaSafeCall(cudaMalloc(&first_APD_d, param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMalloc(&contour_plot, param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMalloc(&stimArea_d, param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMalloc(&stimulus_d,param.memSize));
CudaSafeCall(cudaMalloc(&contour_count_d,sizeof(int)));
CudaSafeCall(cudaMalloc(&contour_vector_d,param.nx*param.ny*sizeof(float3)));

CudaSafeCall(cudaMalloc(&point_d,param.eSize*sizeof(REAL)));

/*------------------------------------------------------------------------
* Set GPU constants
*------------------------------------------------------------------------
*/

CudaSafeCall(cudaMemcpyToSymbol(nx_d, &param.nx, sizeof(int), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(ny_d, &param.ny, sizeof(int), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(rx_d, &param.rx, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(ry_d, &param.ry, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(qx4_d, &param.qx4, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(qy4_d, &param.qy4, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(fx4_d, &param.fx4, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(fy4_d, &param.fy4, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(hx_d, &param.hx, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(hy_d, &param.hy, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(dt_d, &param.dt, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(invdx_d, &param.invdx, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(invdy_d, &param.invdy, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(Lx_d, &param.Lx, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(Ly_d, &param.Ly, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(rxy_d, &param.rxy, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(rbx_d, &param.rbx, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(rby_d, &param.rby, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(rscale_d, &param.rscale, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));

CudaSafeCall(cudaMemcpyToSymbol(boundaryVal_d, &param.boundaryVal, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(solidSwitch_d, &param.solidSwitch, sizeof(bool), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(neumannBC_d, &param.neumannBC, sizeof(bool), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(gateDiff_d, &param.gateDiff, sizeof(bool), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(anisotropy_d, &param.anisotropy, sizeof(bool), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(tipGrad_d, &param.tipGrad, sizeof(bool), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(lap4_d, &param.lap4, sizeof(int), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(timeIntOrder_d, &param.timeIntOrder, sizeof(int), 0,
  cudaMemcpyHostToDevice));

CudaSafeCall(cudaMemcpyToSymbol(tipOffsetX_d, &param.tipOffsetX, sizeof(int), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(tipOffsetY_d, &param.tipOffsetY, sizeof(int), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(minVarColor_d, &param.minVarColor, sizeof(float), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(maxVarColor_d, &param.maxVarColor, sizeof(float), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(tipx0_d, &param.tipx, sizeof(float), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(tipy0_d, &param.tipy, sizeof(float), 0,
  cudaMemcpyHostToDevice));

CudaSafeCall(cudaMemcpyToSymbol(conTh1_d, &param.contourThresh1, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(conTh2_d, &param.contourThresh2, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(conTh3_d, &param.contourThresh3, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(Uth_d, &param.Uth, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));

CudaSafeCall(cudaMemcpyToSymbol(tc_d, &param.tc, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(alpha_d, &param.alpha, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(beta_d, &param.beta, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(gamma_d, &param.gamma, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(delta_d, &param.delta, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(eps_d, &param.eps, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(mu_d, &param.mu, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpyToSymbol(theta_d, &param.theta, sizeof(REAL), 0,
  cudaMemcpyHostToDevice));


printf("Finished allocating device arrays\n");

/*------------------------------------------------------------------------
* Initializing physical arrays. Copy from host to device
*------------------------------------------------------------------------
*/

initGates(pitch,gate_h);

/*------------------------------------------------------------------------
* Initialize device arrays to 0
*------------------------------------------------------------------------
*/

CudaSafeCall(cudaMemset(slice.ux, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice.uy, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice.ut, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice.vx, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice.vy, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice.vt, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.ux, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.uy, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.ut, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.vx, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.vy, 0.0, param.memSize));
CudaSafeCall(cudaMemset(slice0.vt, 0.0, param.memSize));
CudaSafeCall(cudaMemset(advect.x, 0.0, param.memSize));
CudaSafeCall(cudaMemset(advect.y, 0.0, param.memSize));
CudaSafeCall(cudaMemset(velTan.u, 0.0, param.memSize));
CudaSafeCall(cudaMemset(velTan.v, 0.0, param.memSize));
CudaSafeCall(cudaMemset(J_d.u, 0.0, param.memSize));
CudaSafeCall(cudaMemset(J_d.v, 0.0, param.memSize));
CudaSafeCall(cudaMemset(APD1_d, 0.0f, param.memSize));
CudaSafeCall(cudaMemset(APD2_d, 0.0f, param.memSize));

CudaSafeCall(cudaMemset(tip_plot,0,param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMemset(first_APD_d,0,param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMemset(contour_plot,0,param.nx*param.ny*sizeof(bool)));
CudaSafeCall(cudaMemset(contour_vector_d,0.0,param.nx*param.ny*sizeof(float3)));
CudaSafeCall(cudaMemset(tip_vector_d,0.0,TIPVECSIZE*sizeof(vec5dyn)));

printf("Finished initalizing variables\n");
printf("Starting simulation\n\n");

/*------------------------------------------------------------------------
* Load color RGB
*------------------------------------------------------------------------
*/

loadcmap();

/*------------------------------------------------------------------------
* Create masks for domain, stimulus range and measurements
*------------------------------------------------------------------------
*/

domainObjects(solid,coeffTrapz,intglArea,stimArea,
  stimulus,param.stimMag,argc,argv);

/*------------------------------------------------------------------------
* Copy form host to device
*------------------------------------------------------------------------
*/

// Copy data from host to device
CudaSafeCall(cudaMemcpy((void *)gateIn_d.u,(void *)gate_h.u,param.memSize,
  cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy((void *)gateIn_d.v,(void *)gate_h.v,param.memSize,
  cudaMemcpyHostToDevice));

/*------------------------------------------------------------------------
* Copy masks for domain, stimulus range and measurements to device
*------------------------------------------------------------------------
*/

CudaSafeCall(cudaMemcpy(solid_d,solid,param.nx*param.ny*sizeof(bool),cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(coeffTrapz_d,coeffTrapz,param.memSize,cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(intglArea_d,intglArea, param.nx*param.ny*sizeof(bool),cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(stimArea_d,stimArea,param.nx*param.ny*sizeof(bool),cudaMemcpyHostToDevice));
CudaSafeCall(cudaMemcpy(stimulus_d,stimulus, param.memSize, cudaMemcpyHostToDevice));

free(solid);
free(coeffTrapz);
free(intglArea);
free(stimArea);
free(stimulus);
free(cmap_rgba);

/*------------------------------------------------------------------------
* Allocating and initializing OpenGL objects
*------------------------------------------------------------------------
*/

CudaSafeCall(cudaMemcpy((void *)cmap_rgba_data,
                          (void *)cmap_rgba, sizeof(unsigned int)*ncol,
                          cudaMemcpyHostToDevice));

if (false == initGL(&argc, argv)) exit(0);

/*------------------------------------------------------------------------
* Glut loop initialization. Rendering starts here
*------------------------------------------------------------------------
*/

// Return control to the program
glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
//glutCloseFunc(cleanup);
glutMainLoop();

/*------------------------------------------------------------------------
* Copy from device to host
*------------------------------------------------------------------------
*/

CudaSafeCall(cudaMemcpy((void *)gate_h.u,(void *)gateOut_d.u,param.memSize,
  cudaMemcpyDeviceToHost));
CudaSafeCall(cudaMemcpy((void *)gate_h.v,(void *)gateOut_d.v,param.memSize,
  cudaMemcpyDeviceToHost));

/*------------------------------------------------------------------------
* Save data in files
*------------------------------------------------------------------------
*/

if ( param.save && param.savePackage ) {
  saveFile(strAdress,&param,gate_h,electrode,contourLength,param.dt,
    tip_count_d,tip_vector_d,clist,philist,true,contour_count_d,contour_vector_d);
}

/*------------------------------------------------------------------------
* Deallocate all arrays
*------------------------------------------------------------------------
*/

// Free gate host and device memory
free(gate_h.u);
free(gate_h.v);

CudaSafeCall(cudaFree(gateIn_d.u));
CudaSafeCall(cudaFree(gateIn_d.v));
CudaSafeCall(cudaFree(gateOut_d.u));
CudaSafeCall(cudaFree(gateOut_d.v));
CudaSafeCall(cudaFree(cmap_rgba_data));
CudaSafeCall(cudaFree(plot_rgba_data));

CudaSafeCall(cudaFree(slice.ux));
CudaSafeCall(cudaFree(slice.uy));
CudaSafeCall(cudaFree(slice.ut));
CudaSafeCall(cudaFree(slice.vx));
CudaSafeCall(cudaFree(slice.vy));
CudaSafeCall(cudaFree(slice.vt));
CudaSafeCall(cudaFree(slice0.ux));
CudaSafeCall(cudaFree(slice0.uy));
CudaSafeCall(cudaFree(slice0.ut));
CudaSafeCall(cudaFree(slice0.vx));
CudaSafeCall(cudaFree(slice0.vy));
CudaSafeCall(cudaFree(slice0.vt));
CudaSafeCall(cudaFree(advect.x));
CudaSafeCall(cudaFree(advect.y));
CudaSafeCall(cudaFree(velTan.u));
CudaSafeCall(cudaFree(velTan.v));
CudaSafeCall(cudaFree(J_d.u));
CudaSafeCall(cudaFree(J_d.v));

printf("Simulation ended\n");
printf("Physical time: %f ms\n", param.physicalTime);
printf("Real time: %f s\n", param.tiempo);

return 0;

}

void initGates(size_t pitch, stateVar g_h) {

  /*------------------------------------------------------------------------
  * Initialize host arrays
  *------------------------------------------------------------------------
  */

  if ( param.load ) {

    loadData(g_h,strAdress,&param);

  } else {

    int i, j, idx;

    // Array initialization
    memset(g_h.u, 0.0, param.memSize); 
    memset(g_h.v, 0.0, param.memSize);

    // Initial condition
    // for (j=(int)floor(0);j<(int)floor(param.ny);j++) {
    //   for (i=(int)floor(0);i<(int)floor(40);i++) {
    //     idx = i + param.nx*j;
    //     g_h.u[idx] = 1.0;
    //   }
    // }

    // Cross-field stimulation

    for (j=(int)floor(0);j<(int)floor(param.ny);j++) {
      for (i=(int)floor(0);i<(int)floor(param.nx/8);i++) {
        idx = i + param.nx*j;
        g_h.u[idx] = 1.0;
      }
    }

    for (j=(int)floor(param.ny/2);j<(int)floor(param.ny);j++) {
      for (i=(int)floor(0);i<(int)floor(param.nx);i++) {
        idx = i + param.nx*j;
        g_h.v[idx] = 1.0;
      }
    }

  }
}

void domainObjects(bool *solid, REAL *coeffTrapz, bool *intglArea, bool *stimArea,
  REAL *stimulus, REAL stimMag, int argc, char *argv[]) {

  int i, j, idx;

  /*------------------------------------------------------------------------
  * Load unexcitable holes
  * Heterogeneous media
  *------------------------------------------------------------------------
  */

  if ( param.solidSwitch ) {

    float mesh;

    memset(solid, false, param.nx*param.ny*sizeof(bool));

  /*------------------------------------------------------------------------
  * Load domain boundary
  *------------------------------------------------------------------------
  */

    FILE *fp_img;
    char strPath[128];

    if ( param.nx == 512 ) {
      // fp_img = fopen("./common/cBoundary512.dat","r");
      fp_img = fopen("./common/holes512.dat","r");
    }

    if ( param.nx == 800 ) {
      if ( argc>2) {
        sprintf(strPath, "./common/%s",argv[4]);
        fp_img = fopen(strPath,"r");
      } else {
        fp_img = fopen("./common/holes800_1.dat","r");
      }
    }

    if ( param.nx == 1000 ) {
      fp_img = fopen("./common/holes1000.dat","r");
    }

    if ( param.nx == 1024 ) {
      // fp_img = fopen("./common/cBoundary1024.dat","r");
      fp_img = fopen("./common/holes1024.dat","r");
    }

    if ( fp_img==NULL ) {
      printf("Error: can't open circleBoundary file \n");
      exit(0);
    }

    for (i=0;i<param.nx*param.ny;i++){
      fscanf(fp_img, "%f", &mesh);
      // printf("mesh: %d\n", mesh);
      solid[i] = mesh > 0.5 ? true : false;
      // printf("solid %d\n", (int)solid[i]);
    }
    fclose(fp_img);

  }

  /*------------------------------------------------------------------------
  * Slice (derivative) boundary (integration area) 
  *------------------------------------------------------------------------
  */

  // Be sure that the radius of the slice boundary is always smaller than
  // the radius of the domain. If it's not, the derivatives will be 
  // discontinious at the boundary.

  float x0, y0;
  // rdom = 0.5*(Lx-250.0*hx);
  // param.rdomTrapz = 0.5*(param.Lx-(param.tipOffsetX+param.tipOffsetY)*param.hx);
  param.rdomTrapz = 0.5*((param.tipOffsetX+param.tipOffsetY)*param.hx); // Integral area radius

  for (j=0;j<param.ny;j++) {
    for (i=0;i<param.nx;i++) {
      idx = I2D(param.nx,i,j);
      x0 = (float)i*param.hx - 0.5*param.Lx ;
      y0 = (float)j*param.hy - 0.5*param.Ly ;

      intglArea[idx] = ( (x0*x0 + y0*y0) < param.rdomTrapz*param.rdomTrapz ) ? true : false;

    }
  }

  // Points for OpenGL
  for (i=0;i<param.nc;i++) {
    trapzAreaCircle[i].x = (float)( param.nx/2.f+(param.nx-1)/param.Lx*
      param.rdomTrapz*sin((float)i*(2.0*pi)/(param.nc-1)) );
    trapzAreaCircle[i].y = (float)( param.ny/2.f+(param.ny-1)/param.Ly*
      param.rdomTrapz*cos((float)i*(2.0*pi)/(param.nc-1)) );
  }
  

  /*------------------------------------------------------------------------
  * Load trapezoidal coefficients
  *------------------------------------------------------------------------
  */

  REAL *coeffx, *coeffy;
  coeffx = (REAL*)malloc(param.memSize);
  coeffy = (REAL*)malloc(param.memSize);

  memset(coeffx, 0.0, param.memSize);
  memset(coeffy, 0.0, param.memSize);
  memset(coeffTrapz, 0.0, param.nx*param.ny*sizeof(REAL));

  for (i=0;i<param.nx*param.ny;i++) {
    if ( intglArea[i] == true ) {
      coeffx[i] = 2.0;
    }
    if ( intglArea[i] == true && intglArea[i-1] == false ) {
      coeffx[i] = 1.0;
    }
    if ( intglArea[i] == true && intglArea[i+1] == false ) {
      coeffx[i] = 1.0;
    }
  }

  // transpose matrix
  for (j=0;j<param.ny;++j) {
    for (i=0;i<param.nx;++i) {
      coeffy[(i * param.ny) + j] = coeffx[(j * param.nx) + i];
    }
  }

  for (i=0;i<param.nx*param.ny;i++) {
    coeffTrapz[i] = coeffx[i]*coeffy[i]; 
  }

  free(coeffx);
  free(coeffy);

  /*------------------------------------------------------------------------
  * Define the stimulus area
  *------------------------------------------------------------------------
  */

  param.stcx = 0.25*param.Lx; // Stimulus position
  param.stcy = 0.25*param.Ly;
  param.rdomStim = 0.03*param.Lx; // Stimulus radius

  for (j=0;j<param.ny;j++) {
    for (i=0;i<param.nx;i++) {
      idx = I2D(param.nx,i,j);
      x0 = (float)i*param.hx - 0.5*param.Lx ;
      y0 = (float)j*param.hy - 0.5*param.Ly ;

      if ( param.solidSwitch ) {

        // bool sc = solid[idx];
        // bool sw = solid[I2D(param.nx,i-1,j)];
        // bool se = solid[I2D(param.nx,i+1,j)];
        // // Electric field from left to right
        // stimulus[idx] = sc && !sw ?  param.stimMag : 0.0 ; // to the right
        // stimulus[idx] = sc && !se ? -param.stimMag : 0.0 ; // to the left

        // stimulus[idx] = param.stimMag;

        // point stimulus
        stimulus[idx] = ( ((x0-param.stcx)*(x0-param.stcx)+(y0-param.stcy)*(y0-param.stcy)) 
         < param.rdomStim*param.rdomStim ) ? param.stimMag : 0.0;

      } else {

        // For contour stimulation
        // stimulus[idx] = param.stimMag;

        // point stimulus
        stimulus[idx] = ( ((x0-param.stcx)*(x0-param.stcx)+(y0-param.stcy)*(y0-param.stcy)) 
         < param.rdomStim*param.rdomStim ) ? param.stimMag : 0.0;

        // Rectangular stimulus
        // stimulus[idx] = ( (j>20) && (j<30) ) ? param.stimMag : 0.0;
        
      }

    }
  }

  // Point for OpenGL
  param.pointStim.x = param.nx/2+(param.nx-1)/param.Lx*param.stcx;
  param.pointStim.y = param.ny/2+(param.ny-1)/param.Ly*param.stcy;

  /*------------------------------------------------------------------------
  * Define the area arround the stimulus point where we are measureing the APD(x,y) (OPTIONAL)
  *------------------------------------------------------------------------
  */

  for (j=0;j<param.ny;j++) {
    for (i=0;i<param.nx;i++) {
      idx = I2D(param.nx,i,j);
      x0 = (float)i*param.hx - 0.5*param.Lx ;
      y0 = (float)j*param.hy - 0.5*param.Ly ;

      if ( param.solidSwitch ) {

        // bool sc = solid[idx];
        // stimArea[idx] = sc ? true : false;

        // Point stimulus
        stimArea[idx] = ( ((x0-param.stcx)*(x0-param.stcx)+(y0-param.stcy)*(y0-param.stcy)) 
         < param.rdomAPD*param.rdomAPD ) ? false : true;

      } else {

        // stimArea[idx] = true;

      // Point stimulus
      // stimArea[idx] = ( ((x0-param.stcx)*(x0-param.stcx)+(y0-param.stcy)*(y0-param.stcy)) 
      //  < param.rdomAPD*param.rdomAPD ) ? false : true;

      // Rectangular stimulus
      stimArea[idx] = (j<35) ? false : true;

      }

    }
  }

  // Points for OpenGL
  for (i=0;i<param.nc;i++) {
    stimAreaCircle[i].x = param.nx/2+(param.nx-1)/param.Lx*param.stcx+(param.nx-1)/param.Lx*
      param.rdomAPD*sin((float)i*(2.0*pi)/(param.nc-1));
    stimAreaCircle[i].y = param.ny/2+(param.ny-1)/param.Ly*param.stcy+(param.nx-1)/param.Ly*
      param.rdomAPD*cos((float)i*(2.0*pi)/(param.nc-1));
  }

}

/*------------------------------------------------------------------------
* Here is where all the kernel calls and tip tracking, etc. are made
*------------------------------------------------------------------------
*/

// This function is called automatically, over and over again,  by GLUT
void display(void) {

  glutSetWindow(window1);

  if (param.animate) {

    #pragma unroll
    for (int i=0;i<(param.itPerFrame);i++) {

      /*------------------------------------------------------------------------
      * Reaction-diffusion Solver
      *------------------------------------------------------------------------
      */

      if ( !param.reduceSym ) {

        /* Reaction diffusion */
        reactionDiffusion_wrapper(pitch,grid2D,block2D,gateOut_d,gateIn_d,J_d,
          velTan,param.reduceSym,solid_d,stimLock,stimulus_d,stimLockMouse,param.point);

        swapSoA(&gateIn_d,&gateOut_d);

        param.count++;
        param.physicalTime = param.dt*(REAL)param.count;

      } 

      /*------------------------------------------------------------------------
      * Symmetry reduction Solver
      *------------------------------------------------------------------------
      */

      if ( param.reduceSym ) {

        reactionDiffusion_wrapper(pitch,grid2D,block2D,gateOut_d,gateIn_d,J_d,
          velTan,param.reduceSym,solid_d,stimLock,stimulus_d,stimLockMouse,param.point);

        // if ( (param.count%sampleIt == 0) ) {
          tip_wrapper(pitch,grid2D,block2D,gateIn_d,gateOut_d,velTan,param.physicalTime,
            param.tipAlgorithm,param.recordTip,tip_plot,tip_count_d,tip_vector_d);
          clist.push_back(c);
          philist.push_back(phi);
        // }

        slice_wrapper(pitch,grid2D,block2D,gateIn_d,
          slice,slice0,param.reduceSym,param.reduceSymStart,advect,2,intglArea_d,
          tip_count_d,tip_vector_d,param.count);

        if ( param.count == 0 ) {
          trapz_wrapper(grid1D,block2D,slice,slice0,velTan,integrals,coeffTrapz_d,
            tip_count_d,tip_vector_d,param.count);

          c = solve_matrix(c,phi,integrals);

          Cxy_field_wrapper(pitch,grid2D,block2D,advect,c,phi,solid_d);

          slice_wrapper(pitch,grid2D,block2D,gateIn_d,
            slice,slice0,param.reduceSym,param.reduceSymStart,advect,2,intglArea_d,
            tip_count_d,tip_vector_d,param.count);
        }

        trapz_wrapper(grid1D,block1D,slice,slice0,velTan,integrals,coeffTrapz_d
          ,tip_count_d,tip_vector_d,param.count);

        c = solve_matrix(c,phi,integrals);
        // c.y = integrals[10]/integrals[4]; // Translational symmetry in y
        // c.x = integrals[9]/integrals[0]; // Translational symmetry in x

        Cxy_field_wrapper(pitch,grid2D,block2D,advect,c,phi,solid_d);

        advFDBFECC_wrapper(pitch,grid2D,block2D,gateIn_d,gateOut_d,advect,
          uf_d,ub_d,ue_d,solid_d);

        if ( param.reduceSymStart ) {
          phi.x = phi.x+c.x*param.dt;
          phi.y = phi.y+c.y*param.dt;
          phi.t = phi.t+c.t*param.dt;
          c0.x = c.x;
          c0.y = c.y;
          c0.t = c.t;
        } else {
          phi.x = phi.x+param.dt*(1.5*c.x-0.5*c0.x);
          phi.y = phi.y+param.dt*(1.5*c.y-0.5*c0.y);
          phi.t = phi.t+param.dt*(1.5*c.t-0.5*c0.t);
          c0.x = c.x;
          c0.y = c.y;
          c0.t = c.t;
        }

        param.count++;
        param.physicalTime = param.dt*(REAL)param.count;

      }

      /*------------------------------------------------------------------------
      * Special functions
      *------------------------------------------------------------------------
      */

      if ( param.count%sampleIt == 0 ) {
        if ( param.plotTip && !param.reduceSym ) {
          tip_wrapper(pitch,grid2D,block2D,gateIn_d,gateOut_d,velTan,param.physicalTime,
            param.tipAlgorithm,param.recordTip,tip_plot,tip_count_d,tip_vector_d);
        }
        if ( param.recordTip ) {
          saveFile(strAdress,&param,gate_h,electrode,contourLength,param.dt,tip_count_d,tip_vector_d, 
            clist,philist,false,contour_count_d,contour_vector_d);
        }
        if ( param.plotContour ) {
          if ( param.contourMode==1 ) {
            countour_wrapper(pitch,grid2D,block2D,gateOut_d.u,sAPD_d,contour_plot,stimArea_d,
              contour_count_d,contour_vector_d,param.physicalTime,param.contourMode);
          }
          if ( ( param.contourMode==2 ) || ( param.contourMode==3 ) ) {
            countour_wrapper(pitch,grid2D,block2D,gateOut_d.u,gateOut_d.v,contour_plot,stimArea_d,
              contour_count_d,contour_vector_d,param.physicalTime,param.contourMode);
          }
          if ( param.recordContour ) {
            saveFile(strAdress,&param,gate_h,electrode,contourLength,param.dt,tip_count_d,tip_vector_d,
              clist,philist,false,contour_count_d,contour_vector_d);
          }
        }
        if ( param.saveEveryIt && (param.physicalTime>param.startRecTime) ) {
          CudaSafeCall(cudaMemcpy((void *)gate_h.u,(void *)gateOut_d.u,param.memSize,
            cudaMemcpyDeviceToHost));
          saveFile(strAdress,&param,gate_h,electrode,contourLength,param.dt,tip_count_d,tip_vector_d, 
            clist,philist,false,contour_count_d,contour_vector_d);
        }
      }

      // Automatic periodic pacing
      if ( param.stimulate ) {
        if ( (param.count%stimPeriodIt == 0) || (stimCount > 0) ) {
          stimLock = true;
          stimCount++;
          if ( stimCount>stimDurationIt ) {
          	param.leapShocks++;
            stimLock = false;
            stimCount = 0;
          }
        }
      }

      // // Find if the tissue is still excited tissue
      // if ( !isThereFib(timeSeries.u,param) ) {
      //   param.fibTerminated = true;
      //   glutLeaveMainLoop();
      //   glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
      //   printf("No fibrillation. Abort program\n");
      // }

      // Stim with keyboard
      if ( stimLockKey ) {
        stimLock = true;
        stimCount++;
        if ( stimCount>stimDurationIt ) {
          stimLock = false;
          stimCount = 0;
          stimLockKey = false;
        }
      }

      // Stim with mouse
      if ( stimLockMouse ) {
        stimLock = true;
        stimCount++;
        if ( stimCount>stimDurationIt ) {
          stimLock = false;
          stimCount = 0;
          stimLockMouse = false;
        }
      }

      if ( param.contourMode==1 ) {
        sAPD_wrapper(pitch,grid1D,block1D,param.count,gateIn_d.u,gateOut_d.u,APD1_d,APD2_d,sAPD_d,dAPD_d,
          back_d,front_d,first_APD_d,stimArea_d,param.stimulate);
      }

      singleCell_wrapper(pitch,grid0D,block0D,gateOut_d,param.eSize,point_h,point_d,
        param.point);

    } // ITPERFRAME for loop ends

    /*------------------------------------------------------------------------
    * Plot in screen
    *------------------------------------------------------------------------
    */

    // For plotting, map the plot_rgba_data array to the
    // gl_PBO pixel buffer
    CudaSafeCall(cudaGLMapBufferObject((void**)&plot_rgba_data, gl_PBO));

    if ( param.plotTip ) {
      get_rgba_wrapper(pitch,grid2D,block2D,ncol,gateOut_d.u,plot_rgba_data,cmap_rgba_data,tip_plot);
    } else if ( param.plotContour ) {
      get_rgba_wrapper(pitch,grid2D,block2D,ncol,gateOut_d.u,plot_rgba_data,cmap_rgba_data,contour_plot);
    } else {
      get_rgba_wrapper(pitch,grid2D,block2D,ncol,gateOut_d.u,plot_rgba_data,cmap_rgba_data,tip_plot);
    }

    /*------------------------------------------------------------------------
    * Update time series
    *------------------------------------------------------------------------
    */

    int *contour_pts;
    contour_pts = (int*)malloc(sizeof(int));
    CudaSafeCall(cudaMemcpy(contour_pts,contour_count_d,sizeof(int),cudaMemcpyDeviceToHost));

    if ( param.plotTimeSeries ) {

      // Voltage, gates
      #pragma unroll
      for (int i=0;i<(param.wnx-1);i++) {
        timeSeries.u[i] = timeSeries.u[i+1]; // Shift elements to the left
        timeSeries.v[i] = timeSeries.v[i+1];

        // Symmetry reduction
        if ( param.reduceSym ) {
          timeSeriesCx[i] = timeSeriesCx[i+1];
          timeSeriesCy[i] = timeSeriesCy[i+1];
          timeSeriesCt[i] = timeSeriesCt[i+1];
        }
        timeSeriesContourLength[i] = timeSeriesContourLength[i+1];
      }

      // Voltage, gates
      timeSeries.u[param.wnx-1] = point_h[0];
      timeSeries.v[param.wnx-1] = point_h[1];
      // Symmetry reduction
      if ( param.reduceSym ) {
        timeSeriesCx[param.wnx-1] = c.x;
        timeSeriesCy[param.wnx-1] = c.y;
        timeSeriesCt[param.wnx-1] = c.t;
      }

      // Contours
      timeSeriesContourLength[param.wnx-1] = (REAL)*contour_pts;
    }

    if ( param.recordTimeSeries ) {
      if ( param.count%sampleIt == 0 ) {

        // Record single-cell electrodes
        electrodeVar data = {
          point_h[0],
          point_h[1],
        };
        electrode.push_back(data);

        // Record contour length
        contourLength.push_back((REAL)*contour_pts);

      }
    }

    free(contour_pts);

  }

  /*------------------------------------------------------------------------
  * Time limit. Terminate simulation
  *------------------------------------------------------------------------
  */

  if ( param.physicalTime >= (param.physicalTimeLim-1.0) ) {
    glutLeaveMainLoop();
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  }
  
  CudaSafeCall(cudaGLUnmapBufferObject(gl_PBO));

  // Copy the pixel buffer to the texture, ready to display
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,param.nx,param.ny,GL_RGBA,GL_UNSIGNED_BYTE,0);

  // Render one quad to the screen and colour it using our texture
  // i.e. plot our plotvar data to the screen
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_QUADS);
  glTexCoord2f (0.0, 0.0);
  glVertex3f (0.0, 0.0, 0.0);
  glTexCoord2f (1.0, 0.0);
  glVertex3f (param.nx, 0.0, 0.0);
  glTexCoord2f (1.0, 1.0);
  glVertex3f (param.nx, param.ny, 0.0);
  glTexCoord2f (0.0, 1.0);
  glVertex3f (0.0, param.ny, 0.0);
  glEnd();

  /*------------------------------------------------------------------------
  * Add figures to screen
  *------------------------------------------------------------------------
  */

  addFigures(param.point,param.pointStim,trapzAreaCircle,stimAreaCircle,
    param,tip_count_d,tip_vector_d);

  glutSwapBuffers();

  computeFPS();

}

void mouse(int button, int state, int x, int y) {

// GLUT mouse callback. Left button draws the solid, right button removes solid

  float xx,yy;

  if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
    xx = x;
    yy = y;
    param.point.x = xx/width*param.nx;
    param.point.y = (height-yy)/height*param.ny;

  }

  // glutPostRedisplay();
}

void displaySingleCell(void) {

  glutSetWindow(window2);
  glutPostRedisplay();
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();

  /*------------------------------------------------------------------------
  * Time series plot
  *------------------------------------------------------------------------
  */

  if (timeScreen) {

    glTranslatef(0.0,0.1*param.wnx,0.0);

    //////////////////// Grid

    glLineWidth(4.0);
    glColor3f(0.752,0.752, 0.752); // gray
    glPushAttrib(GL_ENABLE_BIT); 
    glLineStipple(1, 0x0F0F);
    glEnable(GL_LINE_STIPPLE);
    glBegin(GL_LINES);
    // Horizontal lines
    glVertex2f(0.0, 0.8*param.wny);
    glVertex2f(param.wnx, 0.8*param.wny);
    glVertex2f(0.0, 0.6*param.wny);
    glVertex2f(param.wnx, 0.6*param.wny);
    glVertex2f(0.0, 0.4*param.wny);
    glVertex2f(param.wnx, 0.4*param.wny);
    glVertex2f(0.0, 0.2*param.wny);
    glVertex2f(param.wnx, 0.2*param.wny);
    glVertex2f(0.0, 0.0);
    glVertex2f(param.wnx, 0.0);
    // Ticks
    glVertex2f(0.2*param.wnx,-10.0);
    glVertex2f(0.2*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.4*param.wnx,-10.0);
    glVertex2f(0.4*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.6*param.wnx,-10.0);
    glVertex2f(0.6*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.8*param.wnx,-10.0);
    glVertex2f(0.8*param.wnx,0.8*param.wny+10.0);

    glEnd();
    glPopAttrib();

    //////////////////// voltage time series

    glLineWidth(2.0);
    glColor3f(0.0,0.0,1.0); // blue
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = 0.8*param.wny*timeSeries.u[i] / abs(param.maxVarColor-param.minVarColor);
      glVertex2d(t,y);
    }
    glEnd();

    glLineWidth(2.0);
    glColor3f(1.0,0.0,0.0); // red
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = 0.8*param.wny*timeSeries.v[i] / abs(param.maxVarColor-param.minVarColor);
      glVertex2d(t,y);
    }
    glEnd();

  }

  /*------------------------------------------------------------------------
  * Phase space plot
  *------------------------------------------------------------------------
  */

  if ( phaseScreen && param.reduceSym ) {

    float cScale = 2.0;

    glTranslatef(0.0,0.5*param.wnx,0.0);

    //////////////////// Grid

    glLineWidth(4.0);
    glColor3f(0.752,0.752, 0.752); // gray
    glPushAttrib(GL_ENABLE_BIT); 
    glLineStipple(1, 0x0F0F);
    glEnable(GL_LINE_STIPPLE);
    glBegin(GL_LINES);

    // Horizontal lines
    glVertex2f(0.0, 0.0);
    glVertex2f(param.wnx, 0.0);
    glVertex2f(0.0, 0.25*param.wny);
    glVertex2f(param.wnx, 0.25*param.wny);
    glVertex2f(0.0, -0.25*param.wny);
    glVertex2f(param.wnx, -0.25*param.wny);

    // Vertical lines
    glVertex2f(0.2*param.wnx,-0.5*param.wnx);
    glVertex2f(0.2*param.wnx,0.5*param.wnx);
    glVertex2f(0.4*param.wnx,-0.5*param.wnx);
    glVertex2f(0.4*param.wnx,0.5*param.wnx);
    glVertex2f(0.6*param.wnx,-0.5*param.wnx);
    glVertex2f(0.6*param.wnx,0.5*param.wnx);
    glVertex2f(0.8*param.wnx,-0.5*param.wnx);
    glVertex2f(0.8*param.wnx,0.5*param.wnx);

    glEnd();
    glPopAttrib();

    //////////////////// Velocity (symmetry reduction) time series

    glLineWidth(2.0);
    glColor3f(0.0,0.0,1.0); // blue
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = param.wny*timeSeriesCx[i]*cScale;
      glVertex2d(t,y);
    }
    glEnd();

    glLineWidth(2.0);
    glColor3f(1.0,0.0,0.0); // red
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = param.wny*timeSeriesCy[i]*cScale;
      glVertex2d(t,y);
    }
    glEnd();

    glLineWidth(2.0);
    glColor3f(0.0,1.0,0.0); // green
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = param.wny*timeSeriesCt[i]*cScale;
      glVertex2d(t,y);
    }
    glEnd();

  }

  if ( contourScreen && param.plotContour ) {

    float cScale = 1.0/param.ny*0.05;

    glTranslatef(0.0,0.1*param.wnx,0.0);

    //////////////////// Grid

    glLineWidth(4.0);
    glColor3f(0.752,0.752, 0.752); // gray
    glPushAttrib(GL_ENABLE_BIT); 
    glLineStipple(1, 0x0F0F);
    glEnable(GL_LINE_STIPPLE);
    glBegin(GL_LINES);
    // Horizontal lines
    glVertex2f(0.0, 0.8*param.wny);
    glVertex2f(param.wnx, 0.8*param.wny);
    glVertex2f(0.0, 0.6*param.wny);
    glVertex2f(param.wnx, 0.6*param.wny);
    glVertex2f(0.0, 0.4*param.wny);
    glVertex2f(param.wnx, 0.4*param.wny);
    glVertex2f(0.0, 0.2*param.wny);
    glVertex2f(param.wnx, 0.2*param.wny);
    glVertex2f(0.0, 0.0);
    glVertex2f(param.wnx, 0.0);
    // Ticks
    glVertex2f(0.2*param.wnx,-10.0);
    glVertex2f(0.2*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.4*param.wnx,-10.0);
    glVertex2f(0.4*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.6*param.wnx,-10.0);
    glVertex2f(0.6*param.wnx,0.8*param.wny+10.0);
    glVertex2f(0.8*param.wnx,-10.0);
    glVertex2f(0.8*param.wnx,0.8*param.wny+10.0);

    glEnd();
    glPopAttrib();

    //////////////////// Contour length series

    glLineWidth(2.0);
    glColor3f(0.0,0.0,1.0); // blue
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINE_STRIP);

    #pragma unroll
    for (int i=0;i<param.wnx;i++) {
      float t = (float)i;
      float y = 0.8*param.wny*timeSeriesContourLength[i]*cScale;
      glVertex2d(t,y);
    }
    glEnd();

  }

  glutSwapBuffers();


}


void resize(int w, int h) {

  // GLUT resize callback to allow us to change the window size

  width = w;
  height = h;
  // glViewport (0, 0,w-param.nx, h);
  glViewport (0, 0, w, h);
  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  glOrtho (0., param.nx, 0., param.ny, -200. ,200.);
  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();


}

void loadcmap(void) {

  /*------------------------------------------------------------------------
  * Load RGB colors
  *------------------------------------------------------------------------
  */

  int i;
  float rcol,gcol,bcol;
  FILE *fp_col;

  //
  // Read in colourmap data for OpenGL display
  //
  fp_col = fopen("./common/yolitzincmap.dat","r");

  if (fp_col==NULL) {
    printf("Error: can't open cmap.dat \n");
    exit(0);
  }

  fscanf (fp_col, "%d", &ncol);
  cmap_rgba = (unsigned int *)malloc(ncol*sizeof(unsigned int));
  CudaSafeCall(cudaMalloc((void **)&cmap_rgba_data, sizeof(unsigned int)*ncol));

  for (i=0;i<ncol;i++) {
    fscanf(fp_col, "%f%f%f", &rcol, &gcol, &bcol);
    cmap_rgba[i]=((int)(255.0f) << 24) | // convert colourmap to int
    ((int)(bcol * 255.0f) << 16) |
    ((int)(gcol * 255.0f) <<  8) |
    ((int)(rcol * 255.0f) <<  0);
  }

  fclose(fp_col);

}

/*------------------------------------------------------------------------
* Graphics stuff
*------------------------------------------------------------------------
*/

int initGL(int *argc, char **argv) {

  //
  // Iinitialise OpenGL display - use glut
  //
  glutInit(argc, argv);

  /*------------------------------------------------------------------------
  * Spiral wave window
  *------------------------------------------------------------------------
  */

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(param.nx, param.ny);                   // Window of nx x ny pixels
  glutInitWindowPosition(800, 50);               // Window position

  window1 = glutCreateWindow("2V-Voltage");         // Window title

  printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
  if(!glewIsSupported(
                      "GL_VERSION_2_0 "
                      "GL_ARB_pixel_buffer_object "
                      "GL_EXT_framebuffer_object "
                      )){
      fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
      fflush(stderr);
      return 1;
  }

  // Set up view
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,param.nx,0.,param.ny, -200.0, 200.0);

  // Create texture and bind to gl_Tex
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);                     // Generate 2D texture
  glBindTexture(GL_TEXTURE_2D, gl_Tex);          // bind to gl_Tex
  // texture properties:
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, param.nx, param.ny, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  printf("Texture created.\n");

  // Create pixel buffer object and bind to gl_PBO
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  unsigned int sizeGL = param.nx*param.ny*sizeof(float); // pitch*param.ny
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeGL, NULL, GL_STREAM_COPY);

  CudaSafeCall( cudaGLRegisterBufferObject(gl_PBO));
  printf("Buffer created.\n");

  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutReshapeFunc(resize);
  glutIdleFunc(display);
  glutTimerFunc(1000/FPS,idle,0); // Timer function will be called after 1000/FPS

  /*------------------------------------------------------------------------
  * Time series window
  *------------------------------------------------------------------------
  */

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(param.wnx, param.wny);                   // Window of nx x ny pixels
  glutInitWindowPosition(800+param.nx+10, 50);               // Window position

  window2 = glutCreateWindow("Time series");         // Window title

  printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
  if(!glewIsSupported(
                      "GL_VERSION_2_0 "
                      "GL_ARB_pixel_buffer_object "
                      "GL_EXT_framebuffer_object "
                      )){
      fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
      fflush(stderr);
      return 1;
  }

  // Set up view
  glClearColor(1.0,1.0,1.0,0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,param.wnx,0.,param.wny, -200.0, 200.0);

  printf("Starting GLUT main loop...\n\n");

  glutKeyboardFunc(keyboard);
  glutReshapeFunc(resize);
  glutDisplayFunc(displaySingleCell);
  // Subwin = glutCreateSubWindow(mainWindow, param.nx, 0, param.nx, param.ny);

  return true;

}

void idle(int) {

  glutPostRedisplay();
  glutTimerFunc(1000/FPS,idle,0); // Timer function will be called after 1000/FPS

}

void cleanup(void) {

  /*------------------------------------------------------------------------
  * Clean, deallocate instructions for after the simulation has ended
  *------------------------------------------------------------------------
  */

  // Free gate host and device memory
  free(gate_h.u); free(gate_h.v);
  CudaSafeCall(cudaFree(gateIn_d.u)); CudaSafeCall(cudaFree(gateIn_d.v));
  CudaSafeCall(cudaFree(gateOut_d.u)); CudaSafeCall(cudaFree(gateOut_d.v));

  puts("\nSimulation ended\n");

  //glBindBuffer(GL_ARRAY_BUFFER, 0);
  //glDeleteBuffers(1, &vbo);

}

__global__ void get_rgba_kernel (size_t pitch, int ncol,
                                 REAL *field,
                                 unsigned int *plot_rgba_data,
                                 unsigned int *cmap_rgba_data,
                                 bool *lines) {

  /*------------------------------------------------------------------------
  * CUDA kernel to fill plot_rgba_data array for plotting
  *------------------------------------------------------------------------
  */

  int icol;
  REAL frac;

  const int i = blockIdx.x*BLOCK_DIM_X + threadIdx.x;
  const int j = blockIdx.y*BLOCK_DIM_Y + threadIdx.y;

  if ( (i<nx_d) && (j<ny_d) ) {

    const int i2d = i + j*nx_d;
    // Change the member of plot_data. to plot a different variable
    frac = (field[i2d]-minVarColor_d)/(maxVarColor_d-minVarColor_d);
    icol = (int)((float)frac*(float)ncol);
    plot_rgba_data[i2d] = (unsigned int)(!lines[i2d]) * cmap_rgba_data[icol];

  }

}

void get_rgba_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, int ncol,
  REAL *field, unsigned int *plot_rgba_data, unsigned int *cmap_rgba_data,
   bool *lines) {

  get_rgba_kernel<<<grid2D,block2D>>>(pitch, ncol,
  	field, plot_rgba_data, cmap_rgba_data, lines); // gateIn_d.u, tip_plot contour_plot
  CudaCheckError();

}

void keyboard(unsigned char key, int x, int y) {

  switch (key) {

    case 'm':
      printf("\n**Keyboard options**\n");
      printf("m --> Menu\n");
      printf("Esc --> Close application\n");
      printf("Space bar --> Pause simulation\n");
      printf("r --> Restart simulation\n");
      printf("q --> Pace/stimulate\n");
      printf("s --> Symmetry reduction\n");
      printf("t --> Tip tracjectory recordings\n");
      printf("c --> Contour recordings\n");
      printf("p --> Print screenshot\n");
      printf("/ --> Conduction block\n");
      printf("x --> Stimulate all tissue once\n");
      printf("x --> Stimulate circle tissue once\n");
      printf("z --> Clear screen\n");
      printf("1 --> Screen 1 (electrodes)\n");
      printf("2 --> Screen 2 (SR velocities)\n");
      printf("3 --> Screen 3 (Contour length)\n");

    break;

    case 27:
      //glutCloseFunc(cleanup);
      // Exit glutMainLoop()
      glutLeaveMainLoop();
      //glutDestroyWindow(glutGetWindow());
      // Return control to the program
      glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_GLUTMAINLOOP_RETURNS);
      printf("Abort program\n");
    break;

    case ' ':
        param.animate = !param.animate;
      if ( param.animate ) {
        printf("unPause\n");
      } else {
        printf("Pause\n");
      }
    break;

    case 'q':
      param.stimulate = !param.stimulate;
      if (param.stimulate) {
        printf("Pacing stimulus\n");
      } else {
        printf("No pacing stimulus\n");
      }
    break;

    case 's':
      printf("clist (symmetry reduction) size %lu\n",clist.size());
      printf("philist (symmetry reduction) size %lu\n",philist.size());
      // Clear memory
      while (!clist.empty()) { 
        clist.pop_back();
      }
      // Clear memory
      while (!philist.empty()) { 
        philist.pop_back();
      }
      param.reduceSym = !param.reduceSym;
      param.reduceSymStart = param.reduceSym;
      if (param.reduceSym) {
        param.dt = 0.5*param.dt;
        param.plotTip = true;
        printf("Symmetry reduction activated\n");
      } else {
        param.dt = 2.0*param.dt;
        param.plotTip = false;
        printf("No symmetry reduction\n");
      }

      if (param.plotTip) {
        printf("Plotting tip trajectory\n");
      } else {
        printf("NOT plotting tip trajectory\n");
      }
    break;

    case 't':
      CudaSafeCall(cudaMemset(tip_count_d,0,sizeof(int))); // Initialize number of contour points
      int *tip_pts;
      tip_pts = (int*)malloc(sizeof(int));
      CudaSafeCall(cudaMemcpy(tip_pts,tip_count_d,sizeof(int),cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemset(tip_vector_d,0,(*tip_pts)*sizeof(vec5dyn)));
      free(tip_pts);
      param.plotTip = !param.plotTip;
      if (param.plotTip) {
        printf("Plotting tip trajectory\n");
      } else {
        printf("NOT recording tip trajectory\n");
      }
    break;

    case 'c':
      CudaSafeCall(cudaMemset(contour_count_d,0,sizeof(int))); // Initialize number of contour points
      int *contour_pts;
      contour_pts = (int*)malloc(sizeof(int));
      CudaSafeCall(cudaMemcpy(contour_pts,contour_count_d,sizeof(int),cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemset(contour_vector_d,0,(*contour_pts)*sizeof(float3)));
      free(contour_pts);
      param.plotContour = !param.plotContour;
      if (param.plotContour) {
        printf("Recording contours\n");
      } else {
        printf("NOT recording contours\n");
      }
    break;

    case 'r':
      initGates(pitch,gate_h);
      // Copy data from host to device
      CudaSafeCall(cudaMemcpy((void *)gateIn_d.u,(void *)gate_h.u,param.memSize,
        cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy((void *)gateIn_d.v,(void *)gate_h.v,param.memSize,
        cudaMemcpyHostToDevice));
      printf("Reset to initial condition\n");
    break;

    case 'p':
      screenShot(param.nx, param.ny);
      printf("Screenshot taken\n");
    break;

    case '/':
      conductionBlock(param.memSize,param.counterclock,param.clock,gate_h,gateIn_d);
      printf("Conduction block applied\n");
    break;

    case 'x':
      stimLockKey = true;
      printf("Stimulating (keyboard) once of %2.1f\n", param.stimMag);
    break;

    case 'o':
      stimLockMouse = true;
	  printf("Stimulating (mouse) once of %2.1f\n", param.stimMag);
	break;

    case 'z':
      CudaSafeCall(cudaMemset(tip_plot,0,param.nx*param.ny*sizeof(bool)));
      CudaSafeCall(cudaMemset(contour_plot,0,param.nx*param.ny*sizeof(bool)));
      printf("Clear screen\n");
    break;

    case '1':
      timeScreen = true;
      phaseScreen = false;
      contourScreen = false;
    break;

    case '2':
      timeScreen = false;
      phaseScreen = true;
      contourScreen = false;
    break;

    case '3':
      timeScreen = false;
      phaseScreen = false;
      contourScreen = true;
    break;

    default:
      puts("No function assigned to this key");
    break;

    }
}

void computeFPS(void) {

  // Count frames per second
  frame_count++;
  final_time = time(NULL);
  if ( (final_time - initial_time) > 0) {
    // frames drawn / time taken (seconds)
    fps = frame_count / (final_time-initial_time);
    // printf("FPS : %d\n", frame_count / (final_time-initial_time));
    frame_count = 0;
    initial_time = final_time;
  }

  GLint64 timer;
  glGetInteger64v(GL_TIMESTAMP, &timer);

  if (param.firstFPS) {
    param.tiempo = 0.0f;
	  base = timer*0.000000001;
	  param.tiempo = timer*0.000000001-base;
    param.firstFPS = false;
    return;
  }

  param.tiempo = timer*0.000000001-base;

  char windowName[256];
  sprintf(windowName, "PhyT %.0f ms | ExcT %.1f s | FPS %d",
    param.physicalTime, param.tiempo, fps);
  glutSetWindowTitle(windowName);
}
