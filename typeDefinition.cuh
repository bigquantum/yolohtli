
#include "globalVariables.cuh"

typedef double REAL;
typedef struct REAL3 { REAL x, y, t; } REAL3;

typedef struct stateVar {
	REAL *u, *v;
} stateVar;

typedef struct advVar {
	REAL *x, *y;
} advVar;

typedef struct sliceVar {
	REAL *ux, *uy, *ut, *vx, *vy, *vt;
} sliceVar;

typedef struct vec5dyn 
	{ float x, y, vx, vy, t; } vec5dyn;

typedef struct electrodeVar {
	REAL e0, e1;
} electrodeVar;

typedef struct velocity {
	REAL x, y;
} velocity;

typedef struct fileVar {
	char read[100], readcsv[100], p1D[100], p2D[100], p2DG[100], tip1[100], tip2[100], sym[100],
		contour1[100], contour2[100], paramFile[100];
} fileVar;

typedef struct paramVar {

	bool animate;
	bool saveEveryIt;
	bool plotTip;
	bool recordTip;
	bool plotContour;
	bool recordContour;
	bool stimulate;
	bool apdContour;
	bool plotTimeSeries;
	bool recordTimeSeries;
	bool reduceSym;
	bool reduceSymStart;
	bool clock, counterclock;
	bool firstIterTip, firstIterContour;
	bool firstFPS;

	bool solidSwitch;
	bool neumannBC;
	bool gateDiff;
	bool anisotropy;
	bool tipGrad;
	int lap4;
	int contourMode;
	int timeIntOrder;
	int tipAlgorithm;

	bool load;
	bool save;

	int nx;
	int ny;
	int memSize;
	REAL Lx, Ly, hx, hy;
	REAL dt;
	REAL diff_par, diff_per;
	REAL Dxx, Dyy, Dxy;
	REAL rx, ry, rxy, rbx, rby;
	REAL rscale;
	REAL invdx, invdy;
	REAL sample;
	int count;
	REAL physicalTime, physicalTimeLim;
	int startRecTime;
	REAL stimPeriod;
	REAL stimDuration;
	REAL stimMag;
	bool fibThreshold;
	bool fibTerminated;
	int leapShocks;
	int eSize;
	int2 point;
	int nc;
	REAL rdomTrapz, rdomStim, rdomAPD;
	REAL stcx, stcy;
	float2 pointStim;
	int savePackage;
	float tiempo;
	REAL degrad;
	REAL boundaryVal;
	REAL qx4, qy4, fx4, fy4;

	int itPerFrame;
	int tipOffsetX;
	int tipOffsetY;
	float minVarColor;
	float maxVarColor;

	int wnx;
	int wny;
	float uMax;
	float uMin;
	float vMax;
	float vMin;

	float tipx;
	float tipy;

	REAL contourThresh1, contourThresh2, contourThresh3;
	REAL Uth;
	REAL tc;
	REAL alpha;
	REAL beta;
	REAL gamma;
	REAL delta;
	REAL eps;
	REAL mu;
	REAL theta;

} paramVar;
