
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include "typeDefinition.cuh"
// #include "globalVariables.cuh"
#include "hostPrototypes.h"

paramVar startMenu(fileVar *strAdress, paramVar param, int argc, char *argv[]) {

	char strPath[128];
	char strDirR[128];
	char strDirRcsv[128];
	char strDirW[128];

	if ( argc>2) {
		sprintf(strPath, "./DATA/");
		sprintf(strDirR, "initCond/data%s/dataSpiral.dat",argv[2]);
		sprintf(strDirRcsv, "initCond/data%s/dataparamcsv.csv",argv[2]);
		sprintf(strDirW, "results/data%s/",argv[3]);

	} else {
		sprintf(strPath, "./DATA/");
		sprintf(strDirR, "initCond/data%s/dataSpiral.dat","A");
		sprintf(strDirRcsv, "initCond/data%s/dataparamcsv.csv","A");
		sprintf(strDirW, "results/data%s/","A");
	}

	/*------------------------------------------------------------------------
	* Directory path
	*------------------------------------------------------------------------
	*/

	memcpy(strAdress->read,strPath,sizeof(strAdress->read));
	memcpy(strAdress->readcsv,strPath,sizeof(strAdress->read));
	memcpy(strAdress->p1D,strPath,sizeof(strAdress->p1D));
	memcpy(strAdress->p2D,strPath,sizeof(strAdress->p2D));
	memcpy(strAdress->p2DG,strPath,sizeof(strAdress->p2DG));
	memcpy(strAdress->tip1,strPath,sizeof(strAdress->tip1));
	memcpy(strAdress->tip2,strPath,sizeof(strAdress->tip2));
	memcpy(strAdress->sym,strPath,sizeof(strAdress->sym));
	memcpy(strAdress->contour1,strPath,sizeof(strAdress->contour1));
	memcpy(strAdress->contour2,strPath,sizeof(strAdress->contour2));
	memcpy(strAdress->paramFile,strPath,sizeof(strAdress->paramFile));

	strcat(strAdress->read,strDirR);
	strcat(strAdress->readcsv,strDirRcsv);
	strcat(strAdress->p1D,strDirW);
	strcat(strAdress->p2D,strDirW);
	strcat(strAdress->p2DG,strDirW);
	strcat(strAdress->tip1,strDirW);
	strcat(strAdress->tip2,strDirW);
	strcat(strAdress->sym,strDirW);
	strcat(strAdress->contour1,strDirW);
	strcat(strAdress->contour2,strDirW);
	strcat(strAdress->paramFile,strDirW);

	printf("LOADING data from %s ?\n",strAdress->readcsv);
	printf("PRESS 'y' to accept, 'n' to start with a default setup, or any key to EXIT program\n");
	char stryesno;
	scanf(" %c", &stryesno);
	// stryesno = 'y'; // Comment after flower garden
	if (stryesno == 'y') {
		printf("Loading parameters\n");

		//////////////////////////////////////
		param.load = false; // load an initial condition or not
		//////////////////////////////////////

		loadParamValues(strAdress->readcsv,&param);

	} else if (stryesno == 'n') {
		printf("Using default setup\n");
		param.load = false;
		param = parameterSetup(param);
    } else {
        printf("EXIT PROGRAM\n");
        exit(1);
    }

	printf("SAVE data to %s ?\n",strAdress->p1D);
	printf("PRESS 'y' to accept, 'n' to decline, or any key to EXIT program\n");
	char stryesno2;
	scanf(" %c", &stryesno2);
	// stryesno2 = 'y'; // Coment after flower garden
	if (stryesno2 == 'y') {
		param.save = true;
	} else if (stryesno2 == 'n') {
		param.save = false;
    } else {
        printf("EXIT PROGRAM\n");
        exit(1);
    }

	return param;


}

paramVar parameterSetup(paramVar param) {

	/*------------------------------------------------------------------------
	* Default switches
	*------------------------------------------------------------------------
	*/

	param.animate = true;
	param.saveEveryIt = false;
	param.plotTip = true;
	param.recordTip = false;
	param.plotContour = false;
	param.recordContour = false;
	param.stimulate = false;
	param.plotTimeSeries = true;
	param.recordTimeSeries = false;
	param.reduceSym = false;
	param.reduceSymStart = param.reduceSym;

	param.solidSwitch = 0;
	param.neumannBC = 1;
	param.gateDiff = 1;
	param.tipAlgorithm = 1; // (1)Bilinear, (2)Newton, (3)Abouzar
	param.anisotropy = 0;
	param.tipGrad = 0;
	param.contourMode = 3;
	param.lap4 = 4 ; // (0) 2nd order, (1) 4th order
	param.timeIntOrder = 4; // (1) Euler, (2) RK2, (4) RK4

	param.clock = 0;
	param.counterclock = 1;
	param.firstFPS = 1;
	param.firstIterTip = true;
	param.firstIterContour = true;

	param.nx = 512;
	param.ny = 512;
	param.Lx = 12.0; // (cm)
	param.Ly = 12.0;
	param.hx = param.Lx/(param.nx-1.0);
	param.hy = param.Ly/(param.ny-1.0);
	param.dt = 0.02; // (ms)
	param.diff_par = 0.001; // (cm^2/ms)
	param.diff_per = 0.001;

	param.degrad = 0.0;
	REAL theta = param.degrad*pi/180.0;
	param.Dxx = param.diff_par*cos(theta)*cos(theta) +
	param.diff_per*sin(theta)*sin(theta);
	param.Dyy = param.diff_par*sin(theta)*sin(theta) +
	param.diff_per*cos(theta)*cos(theta);
	param.Dxy = (param.diff_par - param.diff_per)*sin(theta)*cos(theta);

	// Anisotropic paramters
	param.rxy = 2.0*param.Dxy*param.dt/(4.0*param.hx*param.hy);
	param.rbx = param.hx*param.Dxy/(param.Dxx*param.hy);
	param.rby = param.hy*param.Dxy/(param.Dyy*param.hx);
	param.rx = param.dt*param.Dxx/(param.hx*param.hx);
	param.ry = param.dt*param.Dyy/(param.hy*param.hy);
	param.rscale = 0.01;
	param.invdx = 0.5/param.hx;
	param.invdy = 0.5/param.hy;
	param.qx4 = param.dt*param.Dyy/(param.hy*param.hy*12.0);
	param.qy4 = param.dt*param.Dxx/(param.hx*param.hx*12.0);
	param.fx4 = param.dt/12.0;
	param.fy4 = param.dt/12.0;

	// Global counter and time
	param.count = 0;
	param.physicalTime = 0.0;
	param.physicalTimeLim = 100000.0; // (ms)
	param.startRecTime = 0.0;

	// Single cell recordings
	param.eSize = 2; // Number of electrodes
	param.point = make_int2( param.nx/2, param.ny/2 );

	param.stimPeriod = 600.0; // (ms)
	param.stimMag = 2.0;//(2.0*2000*param.diff_per/param.hx-(-84.0))/(23.125-(-84.0));
	param.stimDuration = 10.0; // (ms)
	param.fibThreshold = 0.1;
	param.fibTerminated = false;
	param.leapShocks = 0;

	param.nc = 100; // Numbre of points for circles
	param.stcx = 0.25*param.Lx; // Stimulus position
  param.stcy = 0.25*param.Ly;
  param.rdomStim = 0.03*param.Lx; // Stimulus area radius
	param.rdomAPD = 0.15*param.Lx; // APD area radius (always greater than rdomStim)
	param.tipOffsetX = 160; // Measured from the center
	param.tipOffsetY = 160; // Measured from the center
	param.rdomTrapz = 0.5*((param.tipOffsetX+param.tipOffsetY)*param.hx);

	param.boundaryVal = 0.0;

	param.itPerFrame = 50;
	param.sample = 2.0; // (ms)

	param.minVarColor = -0.1f;
	param.maxVarColor = 1.1f;
	param.wnx = param.nx;
	param.wny = param.ny;
	param.uMin = -0.1f;
	param.uMax = 1.1f;
	param.vMin = -0.1f;
	param.vMax = 0.5f;

	param.tipx = 0.0f;
	param.tipy = 0.0f;

	param.contourThresh1 = 0.8;
	param.contourThresh2 = 0.85;
	param.contourThresh3 = 0.7;
	param.Uth = 0.7;

	param.tc = 1.0;
	param.alpha = 0.2;
	param.beta = 1.1;
	param.gamma = 0.0;
	param.delta = 1.0;
	param.eps = 0.005;
	param.mu = 1.0;
	param.theta = 0.0;

	return param;

}


fileVar saveFileNames(fileVar strAdress, paramVar *param) {

	/*------------------------------------------------------------------------
	* Create directory
	*------------------------------------------------------------------------
	*/

	DIR* dir = opendir(strAdress.p1D);
	if (dir) {
	    printf("The saving directory already exists.\n");
	    closedir(dir);
	    printf("Do you want to OVERWRITE the directory %s ?\n",strAdress.p1D);
	    printf("PRESS 'y' to accept or any key to EXIT program\n");
	    char stryes;
	    scanf(" %c", &stryes);
	    // stryes = 'y'; // Coment after flower garden
	    if (stryes == 'y') {
	    	int r = remove_directory(strAdress.p1D);
	    	int check;
			check = mkdir( strAdress.p1D , 0700);
	    	if (!check) printf("The directory has been overwritten\n");
	    } else {
	        printf("EXIT PROGRAM\n");
	        exit(1);
	    }
	} else if (ENOENT == errno) {
	    int check;
		check = mkdir( strAdress.p1D , 0700);
		if (!check) printf("NEW directory created: %s\n",strAdress.p1D); 
	} else {
	    /* opendir() failed for some other reason. */
	}

   	/*------------------------------------------------------------------------
	* Select saving package
	*------------------------------------------------------------------------
	*/

	printf("Enter saving Package value:\n");
	printf("--0-No save --1-print2D --2-printTip --3-Contour --4-print2DSubW-multi --5-Symmetry --6-print2DSubW-multi-SR\n");
  
  scanf("%d", &param->savePackage);
  // param->savePackage = 2; // Coment after flower garden

   	/*------------------------------------------------------------------------
	* Choose saving options
	*------------------------------------------------------------------------
	*/

	switch (param->savePackage) {

		case 0:
			printf("You have selected Package %d\n", param->savePackage);
			printf("No files will be saved (except the parameter setup)\n");
			printf("\n");

			param->saveEveryIt = false;
			param->plotTip = false;
			param->recordTip = false;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 1:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -print2D2column\n");
			printf(" -printTip\n");
			printf("\n");

			strcat(strAdress.p2D, "raw_data.dat");

			param->saveEveryIt = false;
			param->plotTip = true;
			param->recordTip = true;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 2:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printTip\n");
			printf("\n");

			strcat(strAdress.tip1, "dataTip.dat");
			strcat(strAdress.tip2, "dataTipSize.dat");

			param->saveEveryIt = false;
			param->plotTip = true;
			param->recordTip = true;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 3:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printContour\n");
			printf("\n");

			strcat(strAdress.contour1, "dataContour.dat");
			strcat(strAdress.contour2, "dataContourSize.dat");

			param->saveEveryIt = false;
			param->plotTip = false;
			param->recordTip = false;
			param->plotContour = true;
			param->recordContour = true;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 4:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -print2DSubWindow ( Multiple it )\n");
			printf("\n");

			param->saveEveryIt = true;
			param->plotTip = false;
			param->recordTip = false;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = false;
			param->reduceSymStart = param->reduceSym;

		break;

		case 5:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -printTip\n");
			printf(" -printSym\n");
			printf("\n");

			strcat(strAdress.tip1, "dataTip_sym.dat");
			strcat(strAdress.tip2, "dataTipSize_sym.dat");
			strcat(strAdress.sym, "c_phi_list_sym.dat");

			param->saveEveryIt = false;
			param->plotTip = true;
			param->recordTip = true;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = true;
			param->reduceSymStart = param->reduceSym;

		break;

		case 6:
			printf("You have selected Package %d\n", param->savePackage);
			printf(" -print2DSubWindow-SR ( Multiple iterations )\n");
			printf("\n");

			param->saveEveryIt = true;
			param->plotTip = false;
			param->recordTip = false;
			param->plotContour = false;
			param->recordContour = false;
			param->stimulate = false;
			param->plotTimeSeries = false;
			param->recordTimeSeries = false;
			param->reduceSym = true;
			param->reduceSymStart = param->reduceSym;

		break;

		default:
      		printf("No function assigned to this key\n");
    	break;

	}

	return strAdress;

}

void saveFile(fileVar strAdress, paramVar *param, stateVar gate_h,
	std::vector<electrodeVar> &electrode, std::vector<REAL> &contourLength,
	int dt, int *tip_count, vec5dyn *tip_vector, std::vector<REAL3> &clist, 
	std::vector<REAL3> &philist, bool saveOnlyLastFrame, int *contour_count,
	float3 *contour_vector) {

   	/*------------------------------------------------------------------------
	* Save data in files
	*------------------------------------------------------------------------
	*/

	char strCount[32];

	switch (param->savePackage) {

		case 0:
			// Do not save anything
		break;

		case 1:
			if ( saveOnlyLastFrame ) {
				print2D2column(strAdress.p2D,gate_h,param);
				saveTipLast(tip_count,tip_vector,param);
			}
		break;

		case 2:
			printTip(strAdress.tip1,strAdress.tip2,tip_count,tip_vector,param);
		break;

		case 3:
			printContour(strAdress.contour1,strAdress.contour2,
				contour_count,contour_vector,param);
		break;

		case 4:
			sprintf(strCount, "raw_data%d.dat", param->count);
			strcat(strAdress.p2D,strCount);
			// print2D2column(strAdress.p2D,gate_h);
			saveTipLast(tip_count,tip_vector,param);
			print2DSubWindow(strAdress.p2D,gate_h,param);
			printf("File %d\n",param->count);
		break;

		case 5:
			printTip(strAdress.tip1,strAdress.tip2,tip_count,tip_vector,param);
			printSym(strAdress.sym,clist,philist);
		break;

		case 6:
			sprintf(strCount, "raw_data_sym%d.dat", param->count);
			strcat(strAdress.p2D,strCount);
			saveTipLast(tip_count,tip_vector,param);
			print2DSubWindow(strAdress.p2D,gate_h,param);
			printf("File %d\n",param->count);
		break;

		default:
      		printf("No function assigned to this key\n");
    	break;

		// print1D(strAdress,gate_h,param);
		// print2DGnuplot(strAdress,gate_h,param);
		// print2D2column(gate_h,count,strAdress,sbytes,param);
		// printVoltageInTime(strAdress,electrode,param);
    // printContourLengthInTime(strAdress,contourLength,param);

	}

	if ( saveOnlyLastFrame ) printParameters(strAdress,param);

}

void loadData(stateVar g_h, fileVar strAdress, paramVar *param) {

  /*------------------------------------------------------------------------
  * Load initial conditions
  *------------------------------------------------------------------------
  */

  int i, j, idx;
  float u, v;

  //Print data
  FILE *fp1;
  fp1 = fopen(strAdress.read,"r");

  if (fp1==NULL) {
    puts("Error: can't open the initial condition file\n");
    exit(0);
  }

  for (j=0;j<param->ny;j++) {
    for (i=0;i<param->nx;i++) {
      idx = i + param->nx * j;
      fscanf(fp1, "%f\t%f", &u, &v);
      g_h.u[idx] = u;
      g_h.v[idx] = v;
    }
  }

  fclose(fp1);

}

void loadParamValues(const char *path, paramVar *param) {

	FILE *fp = fopen(path, "r");
	if (fp == NULL) {
		perror("Unable to open the initial condition csv file");
		exit(1);
	}

	// Count the numbeer of lines
	int ch = 0;
	int nelements = 0;
	while(!feof(fp))
		{
	  	ch = fgetc(fp);
	  	if(ch == '\n') {
	    	nelements++;
	  	}
	}

	float *values;
	values = (float*)malloc(nelements*sizeof(float));

	// Read elements of csv file
	fp = fopen(path, "r");
	int i, l;
	char line[200];

	l = 0;

	while (fgets(line, sizeof(line), fp)) {
		char *token;
		token = strtok(line, ",");

		// Read rows
		for (i=0;i<2;i++) {
			// Read second element of each row
			if (i == 1) {
				values[l] = strtof(token, NULL);
				// printf("%s",token);
			}
			if ( (token != NULL) ) {
				token = strtok(NULL,",");
			}
		}

		l++;
	}

	printf("Elements = %d\n",nelements);

	/*------------------------------------------------------------------------
	* Load parameters
	*------------------------------------------------------------------------
	*/

	// Reading and writing (printing) functions have to have the same order.

	i = 1;
	param->animate = true;
	param->saveEveryIt = (int)values[++i];
	param->plotTip = (int)values[++i];
	param->recordTip = (int)values[++i];
	param->plotContour = (int)values[++i];
	param->recordContour = (int)values[++i];
	param->stimulate = (int)values[++i];
	param->plotTimeSeries = (int)values[++i];
	param->recordTimeSeries = (int)values[++i];
	param->reduceSym = (int)values[++i];
	param->reduceSymStart = (int)param->reduceSym;

	param->solidSwitch = (int)values[++i];
	param->neumannBC = (int)values[++i];
	param->gateDiff = (int)values[++i];
	param->tipAlgorithm = (int)values[++i]; // (1)bilinear, (2)Newton, (3)Abouzar
	param->anisotropy = (int)values[++i];
	param->tipGrad = (int)values[++i];
	param->contourMode = (int)values[++i];
	param->lap4 = (int)values[++i];
	param->timeIntOrder = (int)values[++i];

	param->clock = (int)values[++i];
	param->counterclock = (int)values[++i];

	param->nx = (int)values[++i];
	param->ny = (int)values[++i];
	param->Lx = values[++i];
	param->Ly = values[++i];
	param->hx = values[++i];
	param->hy = values[++i];
	param->dt = values[++i];
	param->diff_par = values[++i];
	param->diff_per = values[++i];

	param->degrad = values[++i]; // Fiber rotation degrees (60)
	param->Dxx = values[++i];
	param->Dyy = values[++i];
	param->Dxy = values[++i];

	// Anisotropic paramters
	param->rxy = values[++i];
	param->rbx = values[++i];
	param->rby = values[++i];
	param->rx = values[++i];
	param->ry = values[++i];
	param->rscale = values[++i];
	param->invdx = values[++i];
	param->invdy = values[++i];
	param->qx4 = values[++i];
	param->qy4 = values[++i];
	param->fx4 = values[++i];
	param->fy4 = values[++i];

	// Global counter and time
	param->count = 0;
	param->physicalTime = 0.0;
	param->physicalTimeLim = values[++i];
	param->startRecTime = (int)values[++i];

	// Single cell recordings
	param->eSize = (int)values[++i]; // Number of electrodes
	param->point = make_int2( (int)values[++i], (int)values[++i] );

	param->stimPeriod = values[++i]; // The number inside is in miliseconds
	param->stimMag = values[++i];
	param->stimDuration = values[++i];
	param->fibThreshold = values[++i];
	param->fibTerminated = values[++i];
	param->leapShocks = values[++i];

	param->nc = (int)values[++i]; // Numbre of points for circles
	param->stcx = values[++i]; // Stimulus position
	param->stcy = values[++i];
	param->rdomStim = values[++i]; // Stimulus area radius
	param->rdomAPD = values[++i]; // APD area radius (always greater than rdomStim)
	param->tipOffsetX = values[++i];
	param->tipOffsetY = values[++i];
	param->rdomTrapz = values[++i];

	param->boundaryVal = values[++i];

	param->itPerFrame = values[++i];
	param->sample = values[++i]; // Sample every 1 second

	param->minVarColor = values[++i];
	param->maxVarColor = values[++i];
	param->wnx = (int)values[++i];
	param->wny = (int)values[++i];
	param->uMin = values[++i];
	param->uMax = values[++i];
	param->vMin = values[++i];
	param->vMax = values[++i];

	param->tipx = values[++i];
	param->tipy = values[++i];

  param->contourThresh1 = values[++i];
  param->contourThresh2 = values[++i];
  param->contourThresh3 = values[++i];
	param->Uth = values[++i];

	param->tc = values[++i];
	param->alpha = values[++i];
	param->beta = values[++i];
	param->gamma = values[++i];
	param->delta = values[++i];
	param->eps = values[++i];
	param->mu = values[++i];
	param->theta = values[++i];

	free(values);

}

// Taken from: https://stackoverflow.com/questions/2256945/removing-a-non-empty-directory-programmatically-in-c-or-c/2256974
int remove_directory(const char *path) {
   DIR *d = opendir(path);
   size_t path_len = strlen(path);
   int r = -1;

   if (d) {
      struct dirent *p;

      r = 0;
      while (!r && (p=readdir(d))) {
          int r2 = -1;
          char *buf;
          size_t len;

          /* Skip the names "." and ".." as we don't want to recurse on them. */
          if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, ".."))
             continue;

          len = path_len + strlen(p->d_name) + 2; 
          buf = (char*)malloc(len);

          if (buf) {
             struct stat statbuf;

             snprintf(buf, len, "%s/%s", path, p->d_name);
             if (!stat(buf, &statbuf)) {
                if (S_ISDIR(statbuf.st_mode))
                   r2 = remove_directory(buf);
                else
                   r2 = unlink(buf);
             }
             free(buf);
          }
          r = r2;
      }
      closedir(d);
   }

   if (!r)
      r = rmdir(path);

   return r;
}