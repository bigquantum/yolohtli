
void loadcmap(void);
int initGL(int *argc, char **argv);
void display(void);
void displaySingleCell(void);
void resize(int w, int h);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void computeFPS(void);
void displayPhaseSpace(void);
void cleanup(void);
void idle(int);

void addFigures(int2 point, float2 pointStim, float2 *trapzAreaCircle,
	float2 *stimAreaCircle, paramVar param,
	int *tip_count, vec5dyn *tip_vector);

//void mouse(int button, int state, int x, int y);
//void mouse_motion(int x, int y);
