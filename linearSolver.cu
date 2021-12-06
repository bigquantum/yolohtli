
/* Driver for routine ludcmp */

#include <stdio.h>
#include <stdlib.h>

#include "typeDefinition.cuh"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"

/*------------------------------------------------------------------------
* Perform LU decomposition
*------------------------------------------------------------------------
*/

void ludcmp(REAL **a, int *indx, REAL *d) {

	int i,imax,j,k;
	REAL big,dum,sum,temp;
	REAL *vv;

	vv=vector(1,NSYM);
	*d=1.0;
	for (i=1;i<=NSYM;i++) {
		big=0.0;
		for (j=1;j<=NSYM;j++)
			if ((temp=fabs(a[i][j])) > big) big=temp;
		if (big == 0.0) nrerror("Singular matrix in routine ludcmp");
		vv[i]=1.0/big;
	}
	for (j=1;j<=NSYM;j++) {
		for (i=1;i<j;i++) {
			sum=a[i][j];
			for (k=1;k<i;k++) sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
		}
		big=0.0;
		for (i=j;i<=NSYM;i++) {
			sum=a[i][j];
			for (k=1;k<j;k++)
				sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
			if ( (dum=vv[i]*fabs(sum)) >= big) {
				big=dum;
				imax=i;
			}
		}
		if (j != imax) {
			for (k=1;k<=NSYM;k++) {
				dum=a[imax][k];
				a[imax][k]=a[j][k];
				a[j][k]=dum;
			}
			*d = -(*d);
			vv[imax]=vv[j];
		}
		indx[j]=imax;
		if (a[j][j] == 0.0) a[j][j]=TINY;
		if (j != NSYM) {
			dum=1.0/(a[j][j]);
			for (i=j+1;i<=NSYM;i++) a[i][j] *= dum;
		}
	}
	
	//free(vv);
	free_vector(vv,1,NSYM);
}

/*------------------------------------------------------------------------
* Solve linear system with LU-decomposition
*------------------------------------------------------------------------
*/

void lubksb(REAL **a, int *indx, REAL b[]) {

	int i,ii=0,ip,j;
	REAL sum;

	for (i=1;i<=NSYM;i++) {
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];
		if (ii)
			for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
		else if (sum) ii=i;
		b[i]=sum;
	}
	for (i=NSYM;i>=1;i--) {
		sum=b[i];
		for (j=i+1;j<=NSYM;j++) sum -= a[i][j]*b[j];
		b[i]=sum/a[i][i];
	}
}

/*------------------------------------------------------------------------
* Numerical recipes allocation and error functions
*------------------------------------------------------------------------
*/

REAL *vector(long nl, long nh)
/* allocate a REAL vector with subscript range v[nl..nh] */
{
	REAL *v;

	v=(REAL *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(REAL)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl+NR_END;
}

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl+NR_END;
}

REAL **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a REAL matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	REAL **m;

	/* allocate pointers to rows */
	m=(REAL **) malloc((size_t)((nrow+NR_END)*sizeof(REAL*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(REAL *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(REAL)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
}

void nrerror(const char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n\n");
	exit(1);
}

void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
	free((FREE_ARG) (m[nrl]+ncl-NR_END));
	free((FREE_ARG) (m+nrl-NR_END));
}