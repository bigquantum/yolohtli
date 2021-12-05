all : solverEx result.txt
solverEx : main.cu reactionDiffusion.cu globalVariables.cuh hostPrototypes.h typeDefinition.cuh \
	devicePrototypes.cuh helper_functions.cu openGLPrototypes.h printFunctions.cu \
	tipTracker.cu integralTrapz.cu symmetryReduction.cu linearSolver.cu singleCell.cu advFDBFECC.cu \
	spaceAPD.cu openGL_functions.cu saveFiles.cu
	nvcc -o solverEx main.cu reactionDiffusion.cu helper_functions.cu printFunctions.cu \
	tipTracker.cu integralTrapz.cu symmetryReduction.cu linearSolver.cu singleCell.cu advFDBFECC.cu \
	spaceAPD.cu openGL_functions.cu saveFiles.cu \
	-arch=sm_61 -lglut -lGL -lGLEW -rdc=true -lSOIL -std=c++11

result.txt : solverEx
	rm -rf result.txt
	./solverEx