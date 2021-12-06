
#include <vector>

paramVar startMenu(fileVar *strAdress, paramVar param, int argc, char *argv[]);
void loadParamValues(const char *path, paramVar *param);
void loadData(stateVar g_h, fileVar strAdress, paramVar *param);
fileVar saveFileNames(fileVar strAdress, paramVar *param);
paramVar parameterSetup(paramVar param);
void saveFile(fileVar strAdress, paramVar *param, stateVar gate_h,
  std::vector<electrodeVar> &electrode, std::vector<REAL> &contourLength,
  int dt, int *tip_count, vec5dyn *tip_vector, std::vector<REAL3> &clist, 
  std::vector<REAL3> &philist, bool saveOnlyLastFrame, int *contour_count,
  float3 *contour_vector);
int remove_directory(const char *path);
void pressEnterKey(void);

void initGates(size_t pitch, stateVar g_h);
void loadcmap(void);
void domainObjects(bool *solid, REAL *coeffTrapz, bool *intglArea, bool *stimArea,
  REAL *stimulus, REAL stimMag, int argc, char *argv[]);

void reactionDiffusion_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
   stateVar gOut_d, stateVar gIn_d, stateVar J,
   stateVar velTan, bool reduceSym, bool *solid, bool stimLock, REAL *stim,
   bool stimLockMouse, int2 point);
void tip_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
   stateVar gOut_d, stateVar gIn_d, stateVar velTan, REAL physicalTime,
   int tipAlgorithm, bool recordTip, bool *tip_plot, int *tip_count, vec5dyn *tip_vector);

void slice_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, 
  stateVar g, sliceVar slice, sliceVar slice0,
  bool reduceSym, bool reduceSymStart, advVar adv, int scheme,
  bool *intglArea, int *tip_count, vec5dyn *tip_vector, int count);
void Cxy_field_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, 
  advVar adv, REAL3 c, REAL3 phi, bool *solid);
void advFDBFECC_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
  stateVar gOut, stateVar gIn,
  advVar adv, stateVar uf, stateVar ub, stateVar ue, bool *solid);
REAL3 solve_matrix(REAL3 c, REAL3 phi, REAL *Int);
void trapz_wrapper(dim3 grid1D, dim3 block1D, sliceVar slice, sliceVar slice0, 
  stateVar velTan, REAL *integrals, REAL *coeffTrapz,
  int *tip_count, vec5dyn *tip_vector, int count);

void singleCell_wrapper(size_t pitch, dim3 grid0D, dim3 block0D, stateVar gOut_d,
  int eSize, REAL *pt_h, REAL *pt_d, int2 point);
void countour_wrapper(size_t pitch, dim3 grid2D, dim3 block2D,
  REAL *field1, REAL *field2, bool *contour_plot, bool *stimArea, int *contour_count, 
  float3 *contour_vector, float physicalTime, int mode);
void sAPD_wrapper(size_t pitch, dim3 grid1D, dim3 block1D, int count, 
  REAL *uold, REAL *unew, REAL *APD1, REAL *APD2, REAL *sAPD, REAL *dAPD, 
  REAL *back, REAL *front, bool *first, bool *stimArea, bool stimulate);
void get_rgba_wrapper(size_t pitch, dim3 grid2D, dim3 block2D, int ncol,
  REAL *field, unsigned int *plot_rba_data, unsigned int *cmap_rgba_data,
   bool *lines);

void swap(float* &a, float* &b);
void swapSoA(stateVar *A, stateVar *B);
float host_lerp(float v0, float v1, float t);
void screenShot(int w, int h);
void conductionBlock(int memSize, bool counterclock, bool clock1,
 stateVar g_h, stateVar g_present_d);
bool isThereFib(REAL *voltage, paramVar param);

void lubksb(REAL **a, int *indx, REAL b[]);
void ludcmp(REAL **a, int *indx, REAL *d);
REAL *vector(long nl, long nh);
int *ivector(long nl, long nh);
REAL **matrix(long nrl, long nrh, long ncl, long nch);
void nrerror(const char error_text[]);
void free_vector(double *v, long nl, long nh);
void free_ivector(int *v, long nl, long nh);
void free_matrix(double **m, long nrl, long nrh, long ncl, long nch);

void print1D(const char *path, stateVar g_h, paramVar *param);
void print2DGnuplot(const char *path, stateVar g_h, paramVar *param);
void printVoltageInTime(const char *path, std::vector<electrodeVar> &sol, 
  paramVar *param);
void printContourLengthInTime(const char *path, std::vector<REAL> &sol, 
  paramVar *param);
void print2D2column(const char *path, stateVar g_h, paramVar *param);
void print2DSubWindow(const char *path, stateVar g_h, paramVar *param);
void printTip(const char *path1, const char *path2, int *tip_count,
  vec5dyn *tip_vector, paramVar *param);
void printContour(const char *path1, const char *path2,
  int *contour_count, float3 *contour_vector, paramVar *param);
void printSym(const char *path, std::vector<REAL3> &clist, std::vector<REAL3> &philist);
void saveTipLast(int *tip_count, vec5dyn *tip_vector, paramVar *param);
void printParameters(fileVar strAdress, paramVar *param);