#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

#include "typeDefinition.cuh"
#include "globalVariables.cuh"

// Function protoypes
#include "openGLPrototypes.h"

#include "./common/CudaSafeCall.h"

/*------------------------------------------------------------------------
* Add OpenGL figures/shapes to screen
*------------------------------------------------------------------------
*/

void addFigures(int2 point, float2 pointStim, float2 *trapzAreaCircle,
	float2 *stimAreaCircle, paramVar param,
	int *tip_count, vec5dyn *tip_vector) {

	/*------------------------------------------------------------------------
	* Draw filled circle at the electrode position
	*------------------------------------------------------------------------
	*/

	glPointSize(20.0);
	glEnable(GL_POINT_SMOOTH);
	// glColor3f(1.0,0.0,0.0); // red
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_DST_COLOR);
	glBegin(GL_POINTS);
	glVertex2f(point.x,point.y);
	glEnd();
	glDisable( GL_BLEND );
	glDisable( GL_POINT_SMOOTH );

	if ( param.reduceSym ) {

		/*------------------------------------------------------------------------
		* Trapezoidal integration domain
		*------------------------------------------------------------------------
		*/

		glLineWidth(2.0);
		// glColor3f(0.0,0.0,1.0); // blue
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND);
		// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBegin(GL_LINE_STRIP);

		// if ( param.recordTip ) {

		int *tip_pts;
		tip_pts = (int*)malloc(sizeof(int));
		CudaSafeCall(cudaMemcpy(tip_pts,tip_count,sizeof(int),cudaMemcpyDeviceToHost));

		if (*tip_pts > TIPVECSIZE ) {
		printf("ERROR: NUMBER OF TIP POINTS EXCEEDS tip_vector SIZE\n");
		exit(0);
		}

		vec5dyn *tip_array;
		tip_array = (vec5dyn*)malloc(sizeof(vec5dyn));
		CudaSafeCall(cudaMemcpy(tip_array,tip_vector+(*tip_pts-1),
			sizeof(vec5dyn),cudaMemcpyDeviceToHost));
		float cx = tip_array[0].x-param.nx/2.f;
		float cy = tip_array[0].y-param.ny/2.f;

		#pragma unroll
		for (int i=0;i<param.nc;i++) {
			glVertex2d(cx+trapzAreaCircle[i].x,cy+trapzAreaCircle[i].y);
		}

		free(tip_pts);
		free(tip_array);


		//   // Plot the circle at the center of the domain
		//   #pragma unroll
		//   for (int i=0;i<param.nc;i++) {
		// 	glVertex2d(trapzAreaCircle[i].x,trapzAreaCircle[i].y);
		//   }
		  
		glEnd();

	}

	if ( param.stimulate ) {

		/*------------------------------------------------------------------------
		* Stimulus/pacing point
		*------------------------------------------------------------------------
		*/

		glPointSize(15.0);
		glEnable(GL_POINT_SMOOTH);
		// glColor3f(1.0,0.0,0.0); // red
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		// glBlendFunc(GL_CONSTANT_COLOR, GL_ONE_MINUS_DST_COLOR);
		glBegin(GL_POINTS);
		glVertex2f(pointStim.x,pointStim.y);
		glEnd();
		glDisable( GL_BLEND );
		glDisable( GL_POINT_SMOOTH );

		/*------------------------------------------------------------------------
		* Stimulus surrounding area
		*------------------------------------------------------------------------
		*/

		glLineWidth(2.0);
		// glColor3f(0.0,0.0,1.0); // blue
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_BLEND);
		// glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBegin(GL_LINE_STRIP);
		#pragma unroll
		for (int i=0;i<param.nc;i++) glVertex2d(stimAreaCircle[i].x,stimAreaCircle[i].y);
		glEnd();
	}

}
