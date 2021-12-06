# Yolohtli
## A cardiac electrophysiology partial differential equation solver

This software is a numeric PDE solver for ionic cardiac models in 2D with real-time graphics. The program has two main solver modes: standard PDE solver, and symmetry-reduction solver for spiral waves.

<img src=images/blood_vessels.png height="500">
<img src=images/tipCompare.png height="500">
<img src=images/tipSYM2.png height="500">

## This software was developerd by: **Hector Augusto Velasco-Perez** @CHAOS Lab@Georgia Institute of Technology

### Special thanks to:
- Dr. Noah DeTal
- Dr. @Flavio Fenton

## Software general decription
This software allows you to solve reaction-diffusion models PDE models with a diffusive coupling in a 2D domain. This software is implemented in CUDA/C.

## Software requirements
- CUDA v7 or higher
- glew.h, glut.h, freeglut.h
- SOIL.h library (sudo apt-get install libsoil-dev)
