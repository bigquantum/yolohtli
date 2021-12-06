## Y

This software isa numerico PDE solver for ionic cardia models in 2D with real-time graphics. The program is has three main functionalities: standard PDE solver, stadard solver with contour tracking for discordant alternans nodes, and a symmetry-reduction solver for spiral waves.

Methods used:
- Solver:
--  For the standar solver the integration is performed with 

<img src=images/Voltagehot3D2.png height="500">
<img src=images/tipCompare.png height="500">
<img src=images/tipSYM2.png height="500">

## This software was developerd by: **Hector Augusto Velasco-Perez** @CHAOS Lab@Georgia Institute of Technology

### Special thanks to:
- Noah DeTal
- Dr. @Flavio Fenton

## Software general decription
This software allows you to solve the Karma (FK) model with a diffusive coupling in a 2D domain with a constant rotating conducting anisotropy. The software allows for input/output files and real time graphics for user interactivity. This software is implemented in CUDA/C.

## Other features
- Time integration: first order explicit Euler method
- Zero-flux boundary conditions and optional periodic boundary conditions at the top and bottom of the domain
- Switch between anisotropic and isotropic tissue. The anisotropy is a constant rotating anisotropy
- Switch between single and double precision

## Software requirements
- CUDA v7 or higher
- glew.h, glut.h, freeglut.h
- SOIL.h library (sudo apt-get install libsoil-dev)

## Software use (in order of appearance)
- To run the eprogram, open a Linux terminal and type `make`
- `globalVariables.cuh`:
     - `nx`,`ny`,`nz`: Grid size
     - `ITPERFRAME` is the number of iterations it computes in the background without rendering an image.
