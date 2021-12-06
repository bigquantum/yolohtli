#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"

#include "./common/CudaSafeCall.h"

// Print a 1D slice of data
void print1D(const char *path, stateVar g_h, paramVar *param) {

	int i, j, idx;

	i = param->nx/2;

	//Print data
	FILE *fp1;
	fp1 = fopen(path,"w+");

	// Notice we are not saving the ghost points
	for (j=0;j<param->ny;j++) {
		idx = i + param->nx * j;
		fprintf(fp1, "%d\t %f\n", j, (float)g_h.u[idx]);
	}

	fclose (fp1);

	printf("1D data file created\n");

}

// Print a 2D slice of data
void print2DGnuplot(const char *path, stateVar g_h, paramVar *param) {

	int i, j, idx;

	//Print data
	FILE *fp1;
	fp1 = fopen(path,"w+");

	// Notice we are not saving the ghost points
	for (j=0;j<param->ny;j++) {
		for (i=0;i<param->nx;i++) {
			idx = i + param->nx * j;
			fprintf(fp1, "%d\t %d\t %f\n", i, j, (float)g_h.u[idx]);
			}
		fprintf(fp1,"\n");
	}

	fclose (fp1);

	printf("2D GNU format data file created\n");

}

// Print a 2D slice of data
void print2D2column(const char *path, stateVar g_h, paramVar *param) {

  int i, j, idx;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  // Notice we are not saving the ghost points
  for (j=0;j<param->ny;j++) {
    for (i=0;i<param->nx;i++) {
      idx = i + param->nx * j;
      fprintf(fp1,"%f %f\n", (float)g_h.u[idx], (float)g_h.v[idx]);
    }
  }

  fclose (fp1);

  printf("2D data file created\n");

}

void print2DSubWindow(const char *path, stateVar g_h, paramVar *param) {

  int i, j, idx;

  int xmin = floor(param->tipx)-param->tipOffsetX-1;
  int xmax = floor(param->tipx)+param->tipOffsetX+1;
  int ymin = floor(param->tipy)-param->tipOffsetY-1;
  int ymax = floor(param->tipy)+param->tipOffsetY+1;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  // Save a window
  for (j=ymin;j<ymax;j++) {
    for (i=xmin;i<xmax;i++) {
      idx = i + param->nx * j;
      fprintf(fp1,"%f %f\n", (float)g_h.u[idx], (float)g_h.v[idx]);
    }
  }

  fclose (fp1);

  printf("2D data file created\n");
}

// Voltage time tracing
void printVoltageInTime(const char *path, std::vector<electrodeVar> &sol, 
  paramVar *param) {

  int i;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  for (i=0;i<sol.size();i++) {
    fprintf(fp1, "%f\t", i*(float)param->dt*param->itPerFrame);
    fprintf(fp1, "%f\t", (float)sol[i].e0);
    fprintf(fp1, "%f\t", (float)sol[i].e1);
    }

  fclose (fp1);

  printf("Voltage time series file created\n");

}

// Voltage time tracing
void printContourLengthInTime(const char *path, std::vector<REAL> &sol, 
  paramVar *param) {

  int i;

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  for (i=0;i<sol.size();i++) {
    fprintf(fp1, "%f\t", i*(float)param->dt*param->itPerFrame);
    fprintf(fp1, "%f\t", (float)sol[i]);
    }

  fclose (fp1);

  printf("Voltage time series file created\n");

}

void printTip(const char *path1, const char *path2, int *tip_count,
  vec5dyn *tip_vector, paramVar *param) {

  // The contour_vector device array is emptyed every iteration.
  // The data is downloaded to a single file.
  // The number of points downloaded in each iteration is also recorded.

  //Print data
  FILE *fp1, *fp2;

  if (param->firstIterTip==true) {
    fp1 = fopen(path1,"w+");
    fp2 = fopen(path2,"w+");
    param->firstIterTip = false;
  } else {
    fp1 = fopen(path1,"a+"); // Write on the same data file
    fp2 = fopen(path2,"a+");
  }

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

  if (*tip_pts > 0) {
    for (size_t i = 0;i<(*tip_pts);i++) {
      fprintf(fp1,"%f %f %f %f %f\n",tip_array[i].x,tip_array[i].y,
        tip_array[i].vx,tip_array[i].vy,tip_array[i].t);
    }
    fprintf(fp2,"%d\n", *tip_pts);
  }

  fclose (fp1);
  fclose (fp2);

  free(tip_pts);
  free(tip_array);

  printf("Tip files created\n");

}

void printContour(const char *path1, const char *path2,
  int *contour_count, float3 *contour_vector, paramVar *param) {

  // The contour_vector device array is emptyed every iteration.
  // The data is downloaded to a single file.
  // The number of points downloaded in each iteration is also recorded.

  //Print data
  FILE *fp1, *fp2;

  if (param->firstIterContour==true) {
    fp1 = fopen(path1,"w+");
    fp2 = fopen(path2,"w+");
    param->firstIterContour = false;
  } else {
    fp1 = fopen(path1,"a+"); // Write on the same data file
    fp2 = fopen(path2,"a+");
  }

  int *contour_pts;
  contour_pts = (int*)malloc(sizeof(int));
  CudaSafeCall(cudaMemcpy(contour_pts,contour_count,sizeof(int),cudaMemcpyDeviceToHost));
  // printf("No. pts = %d\n",*contour_pts);

  if (*contour_pts > param->nx*param->ny ) {
    printf("ERROR: NUMBER OF CONTOUR POINTS EXCEEDS contour_vector SIZE\n");
    exit(0);
  }

  float3 *contour_array;
  contour_array = (float3*)malloc((*contour_pts)*sizeof(float3));
  CudaSafeCall(cudaMemcpy(contour_array,contour_vector,(*contour_pts)*sizeof(float3),cudaMemcpyDeviceToHost));

  if (*contour_pts > 0) {
    for (size_t i = 0;i<(*contour_pts);i++) {
      fprintf(fp1,"%f %f %f\n",contour_array[i].x,contour_array[i].y,contour_array[i].z);
    }
    fprintf(fp2,"%d\n", *contour_pts);
  }

  fclose (fp1);
  fclose (fp2);

  free(contour_pts);
  free(contour_array);

  printf("Contour files created\n");

}

void printSym(const char *path, std::vector<REAL3> &clist, std::vector<REAL3> &philist) {

  //Print data
  FILE *fp1;
  fp1 = fopen(path,"w+");

  for (size_t i=0;i<(clist.size());i++) {
    fprintf(fp1,"%f %f %f %f %f %f\n", 
    	clist[i].x,clist[i].y,clist[i].t,philist[i].x,philist[i].y,philist[i].t);
    }

  fclose (fp1);

  printf("Symmetry files created\n");

}

void printParameters(fileVar strAdress, paramVar *param) {

  // Reading and writing (printing) functions have to have the same order.

  /*------------------------------------------------------------------------
  * Create dat file
  *------------------------------------------------------------------------
  */

  char resultsPath[100];
  char strDirParamCSV[] = "dataparamcsv.csv";
  memcpy(resultsPath,strAdress.paramFile,sizeof(resultsPath));
  strcat(strAdress.paramFile,strDirParamCSV);

  //Print data
  FILE *fp1;
  fp1 = fopen(strAdress.paramFile,"w+");

  /*------------------------------------------------------------------------
  * Create CSV file
  *------------------------------------------------------------------------
  */

  fprintf(fp1,"Initial condition path:,%s\n", param->load ? strAdress.read : "NA");
  fprintf(fp1,"Results file path:,%s\n",  param->save ? resultsPath : "NA");

  fprintf(fp1,"Save every sampling period:,%d\n", param->saveEveryIt ? 1 : 0);
  fprintf(fp1,"plot tip on screen:,%d\n", param->plotTip ? 1 : 0);
  fprintf(fp1,"Save tip in to data file:,%d\n", param->recordTip ? 1 : 0);
  fprintf(fp1,"Plot contours on screen:,%d\n", param->plotContour ? 1 : 0);
  fprintf(fp1,"Save contours to data file:,%d\n", param->recordContour ? 1 : 0);
  fprintf(fp1,"Pacing stimulus:,%d\n", param->stimulate ? 1 : 0);
  fprintf(fp1,"Plot time series:,%d\n", param->plotTimeSeries ? 1 : 0);
  fprintf(fp1,"Record time series:,%d\n", param->recordTimeSeries ? 1 : 0);
  fprintf(fp1,"Reduce symmetry:,%d\n", param->reduceSym ? 1 : 0);
  fprintf(fp1,"Solid boundary:,%d\n", param->solidSwitch ? 1 : 0);
  fprintf(fp1,"Neumann BCs:,%d\n", param->neumannBC ? 1 : 0);
  fprintf(fp1,"Gate diffusion:,%d\n", param->gateDiff ? 1 : 0);
  fprintf(fp1,"Tip trajectory algorithm:, %d\n", param->tipAlgorithm);
  fprintf(fp1,"Anisotropic tisue:,%d\n", param->anisotropy ? 1 : 0);
  fprintf(fp1,"Tip gradient:,%d\n", param->tipGrad ? 1 : 0);
  fprintf(fp1,"Contour mode (space APD or refractory):,%d\n", param->contourMode);
  fprintf(fp1,"Laplacian order:,%d\n", param->lap4 ? 1 : 0);
  fprintf(fp1,"Scheme order for time integration (Euler or RK4):,%d\n", param->timeIntOrder);

  fprintf(fp1,"Conduction block clock:,%d\n", param->clock ? 1 : 0);
  fprintf(fp1,"Conduction block counterclock:,%d\n", param->counterclock ? 1 : 0);

  fprintf(fp1,"# grid points X =,%d\n", param->nx);
  fprintf(fp1,"# grid points Y =,%d\n", param->ny);

  fprintf(fp1,"Physical Lx length,%f\n", (float)param->Lx);
  fprintf(fp1,"Physical Ly length,%f\n", (float)param->Ly);
  fprintf(fp1,"Physical dx,%f\n", (float)param->hx);
  fprintf(fp1,"Physical dy,%f\n", (float)param->hy);

  fprintf(fp1,"Time step:,%f\n", param->reduceSym ? 2.0*(float)param->dt : param->dt);

  fprintf(fp1,"Diffusion parallel component:,%f\n", (float)param->diff_par);
  fprintf(fp1,"Diffusion perpendicular component:,%f\n", (float)param->diff_per);

  fprintf(fp1,"Initial fiber angle:,%f\n", (float)param->degrad);
  fprintf(fp1,"Diffusion Dxx:,%f\n", (float)param->Dxx);
  fprintf(fp1,"Diffusion Dyy:,%f\n", (float)param->Dyy);
  fprintf(fp1,"Diffusion Dxy:,%f\n", (float)param->Dxy);
  fprintf(fp1,"rxy (2*Dxy*dt/(4*dx*dy)):,%f\n", (float)param->rxy);
  fprintf(fp1,"rbx (hx*Dxy/(Dxx*dy)):,%f\n", (float)param->rbx);
  fprintf(fp1,"rby (hy*Dxy/(Dyy*dx)):,%f\n", (float)param->rby);

  fprintf(fp1,"rx (Dxx*dt/(dx*dx)):,%f\n", (float)param->rx);
  fprintf(fp1,"ry (Dyy*dt/(dy*dy)):,%f\n", (float)param->ry);
  fprintf(fp1,"Gate r-scale:,%f\n", (float)param->rscale);
  fprintf(fp1,"invdx (1/(2*hx)),%f\n", (float)param->invdx);
  fprintf(fp1,"invdy (1/(2*hy)),%f\n", (float)param->invdy);
  fprintf(fp1,"qx4 (dt*hx*hx*Dyy/12):,%f\n", (float)param->qx4);
  fprintf(fp1,"qy4 (dt*hy*hx*Dxx/12):,%f\n", (float)param->qy4);
  fprintf(fp1,"fx4 (dt*hx*hx/12):,%f\n", (float)param->fx4);
  fprintf(fp1,"fy4 (dt*hy*hy/12):,%f\n", (float)param->fy4);

  fprintf(fp1,"Physical time limit:,%f\n", (float)param->physicalTimeLim);
  fprintf(fp1,"Start recording time:,%f\n", (float)param->startRecTime);

  fprintf(fp1,"Number of electrodes,%d\n", param->eSize);
  fprintf(fp1,"Electrode position x:,%f\n", (float)param->point.x);
  fprintf(fp1,"Electrode position y:,%f\n", (float)param->point.y);

  fprintf(fp1,"Stimulus period (ms):,%f\n", param->stimPeriod);
  fprintf(fp1,"Stimulus magnitude:,%f\n", (float)param->stimMag);
  fprintf(fp1,"Stimulus duration (ms):,%f\n", (float)param->stimDuration);
  fprintf(fp1,"Voltage threshold for fibrillation:,%f\n", (float)param->fibThreshold);
  fprintf(fp1,"Fibrillation terminated by pacing:,%d\n", param->fibTerminated ? 1 : 0);
  fprintf(fp1,"Number of LEAP shocks applied:,%d\n", param->leapShocks);

  fprintf(fp1,"Number of points in circles:,%d\n",param->nc);
  fprintf(fp1,"Stimulus position x:,%f\n",(float)param->stcx);
  fprintf(fp1,"Stimulus position y:,%f\n",(float)param->stcy);
  fprintf(fp1,"Stimulus area radius:,%f\n",(float)param->rdomStim);
  fprintf(fp1,"APD area radius:,%f\n",(float)param->rdomAPD);
  fprintf(fp1,"Tip offset x,%d\n",param->tipOffsetX);
  fprintf(fp1,"Tip offset y,%d\n",param->tipOffsetY);
  fprintf(fp1,"Integral area radius:,%f\n",(float)param->rdomTrapz);

  fprintf(fp1,"Dirichlet BC value:,%f\n",(float)param->boundaryVal);

  fprintf(fp1,"Iterations per frame:,%d\n", param->itPerFrame);
  fprintf(fp1,"Sampling period (ms):,%f\n", param->sample);

  fprintf(fp1,"Min signal range,%f\n", (float)param->minVarColor);
  fprintf(fp1,"Max signal range,%f\n", (float)param->maxVarColor);
  fprintf(fp1,"Secondary window size x,%d\n", param->wnx);
  fprintf(fp1,"Secondary window size y,%d\n", param->wny);
  fprintf(fp1,"Secondary window min signal 1,%f\n", (float)param->uMin);
  fprintf(fp1,"Secondary window max signal 1,%f\n", (float)param->uMax);
  fprintf(fp1,"Secondary window min signal 2,%f\n", (float)param->vMin);
  fprintf(fp1,"Secondary window max signal 2,%f\n", (float)param->vMax);

  fprintf(fp1,"Last tip point X,%f\n", (float)param->tipx);
  fprintf(fp1,"Last tip point Y,%f\n", (float)param->tipy);

  fprintf(fp1,"Contour threshold 1:,%f\n", (float)param->contourThresh1);
  fprintf(fp1,"Contour threshold 2:,%f\n", (float)param->contourThresh2);
  fprintf(fp1,"Contour threshold 3:,%f\n", (float)param->contourThresh3);
  fprintf(fp1,"Filament voltage threshold:,%f\n", (float)param->Uth);

  fprintf(fp1,"time scale (tc):,%f\n", (float)param->tc);
  fprintf(fp1,"alpha:,%f\n", (float)param->alpha);
  fprintf(fp1,"beta:,%f\n", (float)param->beta);
  fprintf(fp1,"gamma:,%f\n", (float)param->gamma);
  fprintf(fp1,"delta:,%f\n", (float)param->delta);
  fprintf(fp1,"epsilon:,%f\n", (float)param->eps);
  fprintf(fp1,"mu:,%f\n", (float)param->mu);
  fprintf(fp1,"theta:,%f\n", (float)param->theta);

  fclose (fp1);

  printf("Parameter files created\n");

}