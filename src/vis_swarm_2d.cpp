/*********************************************
 * Adapted from:
 * ------------------
 * Sunlight
 *
 * Author:
 * Susanne Kroemker
 * IWR - UNIVERSITAET HEIDELBERG
 * Im Neuenheimer Feld 205
 * D-69120 Heidelberg
 *
 * phone +49 (0)6221 54 14413
 *
 * EMail  kroemker@iwr.uni-heidelberg.de
 * ------------------
 * (See ../Sunlight/)
 * Author:
 * Marvin Koss
 *********************************************/

#include <stdio.h>
#include <stdlib.h>

using namespace std;
// std includes
#include <iostream>
#include <math.h>
#include <random>
#include <string>

#include "util/arcball.h"
#include "optim/swarm.h" // contains swarm optimizer implementation

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define KEY_ESC 27

const char* funcName = "wellblech"; // NOTE: ADJUSTABLE PARAMETER

// global variables for handling the arcball

// screen size
int screenWidth = 1920;
int screenHeight = 1080;

// last and current y mouse position for zooming
int last_y = 0;
int cur_y = 0;

// arcball activated or not
bool arcball_on = false;

// zoom activated or not
bool zoom_on = false;
float zoomValue = 0.01f;

// initialize arcball 
static gl::ArcBall arcball;

// initialize some colors
static const GLfloat edge_color[] = {0.6f, 0.6f, 0.6f, 1.0f};
static const GLfloat back_color[] = {0.0,0.0,1.0,0.5};
static const GLfloat front_color[] = {1.0,1.0,0.1,0.5 };
static const GLfloat front_triangle_color[] = { 0.6,0.6,0.6,1.0 };
static const GLfloat back_triangle_color[] = { 0.1,0.1,0.15,0.4 };
static const GLfloat plane_color[] = { 0.1,0.1,0.15,0.4 };
// static const GLfloat prtcl_sphere_color[] = { 0.7,0.7,0.99,1.0 };
static const GLfloat outer_sphere_color[] = { 0.0,1.0,0.0,1.0 };
static const GLfloat semi_transparent_back_color[] = { 0.0,0.0,1.0,0.5};
static const GLfloat semi_transparent_front_color[] = { 1.0,0.0,0.0,0.5 };

//colors for arrows
static const GLfloat gbest_arrow_color[] = {1.0, 0.0, 0.0, 1.0};
static const GLfloat pbest_arrow_color[] = {0.0, 0.0, 1.0, 1.0};
static const GLfloat particle_arrow_color[] = {0.0, 1.0, 0.0, 1.0};


// whether to visualize; must be in 2D (DIMS == 2)
static const bool VIZ = true; // TODO implement

/* begin function plot parameter setup */
static const int DIMS = 2; // TODO make 1-D vis possible

// const GLfloat grid_size = 20.0f; // NOTE: ADJUSTABLE PARAMETER
const GLfloat grid_size_x = 150.0f; // NOTE: ADJUSTABLE PARAMETER
const GLfloat grid_size_y = 150.0f; // NOTE: ADJUSTABLE PARAMETER
const GLfloat Xmin[] = {-grid_size_x,0.0f, 0.0f, 0.0f};
const GLfloat Xmax[] = {grid_size_x, 0.0f, 0.0f, 0.0f};
const GLfloat Ymin[] = {0.0f, -grid_size_y, 0.0f, 0.0f};
const GLfloat Ymax[] = {0.0f, grid_size_y, 0.0f, 0.0f};

static const GLfloat tile_width_x = 4.0; // NOTE: ADJUSTABLE PARAMETER
static const GLfloat tile_width_y = 4.0; // NOTE: ADJUSTABLE PARAMETER

const int num_tiles_x = (Xmax[0] - Xmin[0])/tile_width_x;
const int num_tiles_y = (Ymax[1] - Ymin[1])/tile_width_y;

GLfloat zs[num_tiles_x + 1][num_tiles_y + 1]; // will hold z value for each grid vertex
GLfloat v0[DIMS+1]; 
GLfloat v1[DIMS+1]; 
GLfloat mid[DIMS+1]; 
GLfloat lerp;
GLfloat x;
GLfloat y;

/* end function plot parameter setup */

// SWARMOPTIMIZER variables
static const int N_PARTICLES = 10;
static const int N_GROUPS = 1;
GLfloat prtcl_sphere_color[N_GROUPS][4];

// initialize viz toggles
bool plane_toggle = false;
bool gbest_toggle = false;
bool pbest_toggle = false;

int prtcl_group(int i) {
    // takes index i out of N_particles and
    // returns index g out of N_GROUPS
    // that particle i belongs to.
    return int(i/int(N_PARTICLES/N_GROUPS));
}

// particle colors
void set_prtcl_colors(int n_groups) {
    uniform_real_distribution<float> color_dist(0.0, 1.0);
    random_device rd_color;
    mt19937 color_generator(rd_color());
    for (int g=0; g < n_groups; g++) {
        for (int d=0; d < 3; d++) {
            prtcl_sphere_color[g][d] = color_dist(color_generator);
        }
        prtcl_sphere_color[g][3] = 1.0;
    }
}

// function declarations
void draw_fn();
void draw_xy_plane_if_toggled();
void draw_best_if_toggled();
void draw_light();
void draw_coordinate_system( float unit );
void draw_particles();
void display();
void reshape( GLint width, GLint height );
void keyboard( GLubyte key, GLint x, GLint y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void init_glut();
int main( int argc, char** argv );
void set_zs();
GLfloat objective (GLfloat X[DIMS]);

// initialise swarmoptimizer here
// because some function definitions refer to it
// TODO make optim an argument of these
SWARMOPTIMIZER<N_PARTICLES, DIMS, float> optimizer;

/* function definitions */

GLfloat objective (GLfloat X[DIMS]) {
    GLfloat x = X[0];
    GLfloat y = X[1];
    if (funcName == "rastrigin") {
        GLfloat x_comp = x*x - 10 * cos(2.0 * M_PI * x) + 10;
        GLfloat y_comp = y*y - 10 * cos(2.0 * M_PI * y) + 10;
        return x_comp + y_comp;
    } else if (funcName == "wellblech") {
        GLfloat wobble = 4.0f;
        return wobble*sin(x) - wobble*cos(y) + 0.004*pow(y+50, 2) + 0.001*pow(x+50, 2);
    } else if (funcName == "wobble") {
        GLfloat wobble = 8.0f;
        GLfloat r = wobble*sin(x) - wobble*cos(y) + exp(y) + pow(0.2*(x-5.0), 2.0); // -pow(0.05*x, 6.0);
        if (y < -80.0f) r = (80.0f-(y+80.0f));
        return r;
    } else if (funcName == "schaffersf6") {
        GLfloat denominator = pow((sin(sqrt(x*x + y*y))), 2.0) - 0.5;
        GLfloat numerator = pow(1.0 - 0.001 * (x*x+y*y), 2);
        GLfloat frac = denominator/numerator;
        return - 2.0 - frac - 0.1 * (x + y);
    } else if (funcName == "x2") {
        GLfloat lower, upper, r;
        GLfloat wobble = 8.0f;
        r = 0.02 * pow(x,2.0) + sin(y) + 0.01 *pow(y,2.0) - 20.0 + wobble * sin(y);
        lower = 55.0;
        upper = 75.0;
        if (lower < y and y < upper) {
            r -= 100.0;
        }
        return r;
    } else if (funcName == "ripple") {
        return sin(10*(x*x+y*y))/10.0;
    } else if (funcName == "alpine") {
        GLfloat thresh = 0.001;
        return (abs(x*sin(x)+0.1*x) - abs(y*sin(y)+0.1*y) + y*y * 0.01) * 0.1 + 1/max(abs(thresh*x)+abs(thresh*y), thresh);
    } else if (funcName == "alpine0") {
        GLfloat shiftx = 20;
        GLfloat shifty = 20;
        GLfloat shiftz = -10;
        GLfloat thresh = 100;
        if (shiftx * 3 < x && shiftx *5 > x && shifty*3 < y && shifty*7 > y) {
            // be locally repeating but globally increasing
            // x = int(x - shiftx) % 80 + 20;
            // y = int(y + shifty) % 80 + 30;
            return -(x-shiftx*5 + y-shifty*5);
        }
        return (abs(x*sin(x)+0.1*x) + abs(y*cos(y)+0.2*y) + (x*x+y*y) * 0.01 -(x+y)*0.8) * 0.2 - shiftz;
    } else if (funcName == "x2y2") {
        return 0.03 * (x*x + y*y) - 10.0;
    } else if (funcName == "xy") {
        return 0.03 * x*y + 0.0001* pow(x*y,2.0);
    } else if (funcName == "x3y2") {
        GLfloat x5 = 0.000001 * pow(x,4);
        return x5 + 0.01 * (0.1*(x*x*x) + y*y) + atan(y) + 0.0001*(y*y*y*y);
    } else if (funcName == "surprise") {
        GLfloat clamp_val = 3.0;
        GLfloat minx = min(abs(x), clamp_val);
        GLfloat r = pow(minx,2);
        GLfloat step_size = 10.0;
        GLfloat step_size_z = 10.0;

        int tmp = x / step_size; 
        r -= tmp * step_size_z;
        return r;
    } else if (funcName == "abscos") {
        GLfloat a = abs(x) + abs(y);
        return cos(a) * a * 0.00002 * pow(x,2.0) * pow(y,2.0);
    } else if (funcName == "peaks") {
        GLfloat a = 3.0*pow(1.0-x,2)*exp(-pow(x,2.0)-pow(y+1.0, 2.0));
        GLfloat b = 10.0*(x/5.0-pow(x,3.0)-pow(y,5.0))*exp(-(pow(x,2.0)+pow(y,2.0)));
        GLfloat c = 1/3 * exp(-pow(x+1.0,2.0)-pow(y,2.0));
        return a+b+c;
    }
    return 0.0;
}

void set_zs() {
    // get loss values at vertices of triangle grid
    GLfloat x = Xmin[0];
    GLfloat y;
    GLfloat z;
    for (int i = 0; i <= num_tiles_x; i++) {
        y = Ymin[1]; // reset y
        for (int j = 0; j <= num_tiles_y; j++) {
            GLfloat X[2] = {x,y};
            z = objective(X);
            zs[i][j] = z;
            y += tile_width_y;
        }
        x += tile_width_x;
    }
}

void draw_fn() {
    // TODO (low priority): generalize to D dimensions

    // now go over vertices again, splitting each square tile into four equilateral triangles facing inwards
    // using the height values we already got
    x = Xmin[0];
    y;
    for (int i = 0; i < num_tiles_x; i++) {
        y = Ymin[1]; // reset y
        for (int j = 0; j < num_tiles_y; j++) {

            lerp = (zs[i][j] + zs[i+1][j] + zs[i][j+1] + zs[i+1][j+1]) / 4.0f; // midpoint height is linear interpolation of four neighboring vertices
            mid[0] = x + (tile_width_x/2.0f);
            mid[1] = y + (tile_width_y/2.0f);
            mid[2] = lerp;

            for (int x_vertices = 0; x_vertices < 2; x_vertices++) {
                for (int y_vertices = 0; y_vertices < 2; y_vertices++) {
                    // define outer corner points of triangle
                    v0[0] = x + x_vertices * tile_width_x;
                    v0[1] = y + y_vertices * tile_width_y;
                    v0[2] = zs[i + x_vertices][j + y_vertices];
                    // go around counter-clockwise, starting from bottom
                    if ((x_vertices + y_vertices) % 2 == 0) {
                        v1[0] = x + (1 - x_vertices) * tile_width_x;
                        v1[1] = y + y_vertices * tile_width_y;
                        v1[2] = zs[i + (1-x_vertices)][j + y_vertices];
                    } else {
                        v1[0] = x + x_vertices * tile_width_x;
                        v1[1] = y + (1-y_vertices) * tile_width_y;
                        v1[2] = zs[i + x_vertices][j + (1-y_vertices)];
                    }

                    // define corresponding triangle sides (vectors pointing from outer corner toward mid)
                    // (used for normal vector calculation)
                    // const GLfloat s0[] = {mid[0] - v0[0], mid[1] - v0[1], mid[2] - v0[2]};
                    // const GLfloat s1[] = {mid[0] - v1[0], mid[1] - v1[1], mid[2] - v1[2]};
                    //

                    // set triangle color
                    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, back_triangle_color);
                    glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, front_triangle_color);

                    // set triangle
                    glBegin(GL_TRIANGLES);
                    // glNormal3f(s0[1]*s1[2] - s0[2] * s1[1], s0[2]*s1[0] - s0[0] * s1[2], s0[0] * s1[1] - s0[1] * s1[0]); // normal vector is cross product of triangle sides
                    glVertex3f(v0[0],  v0[1],  v0[2]);
                    glVertex3f(mid[0], mid[1], mid[2]);
                    glVertex3f(v1[0],  v1[1],  v1[2]);
                    glEnd();

                    // glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, edge_color);

                    // also draw triangle edges for better visibility NOTE TODO add shadows!!
                    glEnable(GL_LINE_SMOOTH);
                    glBegin(GL_LINES);
                    // edge 1:
                    glVertex3f(v0[0],  v0[1],  v0[2]);
                    glVertex3f(mid[0], mid[1], mid[2]);
                    // edge 2:
                    glVertex3f(mid[0], mid[1], mid[2]);
                    glVertex3f(v1[0],  v1[1],  v1[2]);
                    // edge 3:
                    glVertex3f(v1[0],  v1[1],  v1[2]);
                    glVertex3f(v0[0],  v0[1],  v0[2]);
                    glEnd();

                    // NOTE DEBUG:
                    // if (i==2 && j == 3 ) { // && x_vertices == 0 && y_vertices == 1) {
                    //     cout << "set triangle with " << endl;
                    //     cout << "mid = " <<  mid[0] << ", " << mid[1] << ", " << mid[2] << endl;
                    //     cout << "v0  = " <<  v0[0] << ", " << v0[1] << ", " << v0[2] << endl;
                    //     cout << "v1  = " <<  v1[0] << ", " << v1[1] << ", " << v1[2] << endl;
                    //     cout << "i   = " << i << endl;
                    //     cout << "j   = " << j << endl;
                    //     cout << "x   = " << x << endl;
                    //     cout << "y   = " << y << endl;
                    //     cout << "x_v = " << x_vertices << endl;
                    //     cout << "y_v = " << y_vertices << endl;
                    // }
                }
            }
            y += tile_width_y;
        }
        x += tile_width_x;
    }
    // this couldve been done without the second loop, but seems less convoluted of an implementation
}


void draw_best_if_toggled(){
    int i = N_PARTICLES-1; // Particle to track
    if (gbest_toggle) {
        // draw arrow above best point

        // GLfloat* gbest = optimizer.gbest;
        // static const GLfloat gbest[] = {0.0,0.0,0.0};

        static const GLfloat arrow_offset_z = 5.0;
        static const GLfloat arrow_len = 20.0;

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, gbest_arrow_color);
        glEnable(GL_LINE_SMOOTH);

        glBegin(GL_LINES);
        glVertex3f( optimizer.gbest[0], optimizer.gbest[1], optimizer.gbest[2] + arrow_offset_z + arrow_len);
        glVertex3f( optimizer.gbest[0], optimizer.gbest[1], optimizer.gbest[2] + arrow_offset_z);
        glEnd();

        glPushMatrix();
        glTranslatef(optimizer.gbest[0], optimizer.gbest[1], optimizer.gbest[2] + arrow_offset_z);
        glutSolidCone(0.5, -2.0, 20, 20);
        glPopMatrix();

        glRasterPos3f(optimizer.gbest[0], optimizer.gbest[1], optimizer.gbest[2] + arrow_offset_z - 1.0);
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'G');
    }
    if (pbest_toggle) {
        // draw arrow above best point

        // GLfloat* gbest = optimizer.gbest;
        // static const GLfloat gbest[] = {0.0,0.0,0.0};


        static const GLfloat arrow_offset_z = 5.0;
        static const GLfloat arrow_len = 5.0;

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pbest_arrow_color);
        glEnable(GL_LINE_SMOOTH);

        glBegin(GL_LINES);
        glVertex3f( optimizer.pbests[i][0], optimizer.pbests[i][1], optimizer.pbests[i][2] + arrow_offset_z + arrow_len);
        glVertex3f( optimizer.pbests[i][0], optimizer.pbests[i][1], optimizer.pbests[i][2] + arrow_offset_z);
        glEnd();

        glPushMatrix();
        glTranslatef(optimizer.pbests[i][0], optimizer.pbests[i][1], optimizer.pbests[i][2] + arrow_offset_z);
        glutSolidCone(0.5, -2.0, 20, 20);
        glPopMatrix();

        glRasterPos3f(optimizer.pbests[i][0], optimizer.pbests[i][1], optimizer.pbests[i][2] + arrow_offset_z - 1.0);
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'P');
    }
}

void draw_xy_plane_if_toggled(){
    if (plane_toggle) {
        GLfloat plane_z = 0.0;
        
        // set triangle color
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, plane_color);

        // set triangle
        glBegin(GL_TRIANGLES);
        glVertex3f(-grid_size_x, -grid_size_y, plane_z);
        glVertex3f(grid_size_x, -grid_size_y, plane_z);
        glVertex3f(-grid_size_x, grid_size_y, plane_z);

        glVertex3f(-grid_size_x, grid_size_y, plane_z);
        glVertex3f(grid_size_x, -grid_size_y, plane_z);
        glVertex3f(grid_size_x, grid_size_y, plane_z);
        glEnd();
    }
}


void draw_light()
{
    // Several different light positions, only one should be activated
    // Parallel light from infinity with w = 0.0
    static const GLfloat light_pos[]        = {0.0f, 0.0f, 500.0f, 0.0f};
    //static const GLfloat light_pos[]        = {0.0f, 0.0f, 0.001f, 0.0f};
    //static const GLfloat light_pos[]        = {0.0f, 0.0f, -0.001f, 0.0f};

    // Pointlight from a certain position in space with w = 1.0
    //static const GLfloat light_pos[]        = {1.0f, 1.0f, -4.0f, 1.0f};
    //static const GLfloat light_pos[]        = {0.0f, 0.0f, 3.0f, 1.0f};

    static const GLfloat sun_color[]        = {1.0f, 1.0f, 0.0f, 1.0f};
    static const GLfloat sun_emissive[]     = {1.0f, 1.0f, 1.0f, 1.0f};
    static const GLfloat not_emissive[]     = {0.0f, 0.0f, 0.0f, 1.0f};

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPushMatrix();
    glTranslatef(light_pos[0], light_pos[1], light_pos[2]);

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, sun_color);
    glMaterialfv(GL_FRONT, GL_EMISSION, sun_emissive);
    glutSolidSphere(10.0f, 20, 20);

    glMaterialfv(GL_FRONT, GL_EMISSION, not_emissive);
    glPopMatrix();
}



void draw_coordinate_system(float unit)
{

    static const GLfloat axis_color[] = {0.5f, 0.5f, 0.5f, 1.0f};

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, axis_color);

    glEnable(GL_LINE_SMOOTH);

    glBegin(GL_LINES);
    glVertex3f( 0.0f, 0.0f, 0.0f);
    glVertex3f( 1.0f * unit, 0.0f, 0.0f);
    glVertex3f( 0.0f, 0.0f, 0.0f);
    glVertex3f( 0.0f, 1.0f * unit, 0.0f);
    glVertex3f( 0.0f, 0.0f, 0.0f);
    glVertex3f( 0.0f, 0.0f, 1.0f * unit);
    glEnd();

    glPushMatrix();
    glRotatef(90.0f,0.0,1.0,0.0);
    glTranslatef(0.0f,0.0f,1.0f * unit);
    glutSolidCone(0.1 * unit,0.5 * unit, 20, 20);
    glPopMatrix();
    glPushMatrix();
    glRotatef(270.0f,1.0,0.0,0.0);
    glTranslatef(0.0f,0.0f,1.0f * unit);
    glutSolidCone(0.1 * unit,0.5 * unit, 20, 20);
    glPopMatrix();
    glPushMatrix();
    glTranslatef(0.0f,0.0f,1.0f * unit);
    glutSolidCone(0.1 * unit,0.5 * unit, 20, 20);
    glPopMatrix();

    glRasterPos3f( 1.9, 0.0, 0.0);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'x');

    glRasterPos3f( 0.0, 1.9, 0.0);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'y');

    glRasterPos3f( 0.0, 0.0, 1.9);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'z');
}


void draw_particles()
{
  GLUquadricObj *qobj = gluNewQuadric();
  float sphere_radius = 4.0;

  for (int i = 0; i < N_PARTICLES; i++) {
      int g = prtcl_group(i);
      glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, prtcl_sphere_color[g]);

      glPushMatrix();
      glTranslatef(optimizer.x[i][0], optimizer.x[i][1], optimizer.z[i]);
      gluSphere(qobj,sphere_radius,50,50);
      glPopMatrix();

  }
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluLookAt(0, 0, 8 + zoomValue, 0, 0, 0, 0, 1, 0);

  // multiply current matrix with arcball matrix
  glMultMatrixf(arcball.get());

  draw_coordinate_system(1.0f);
  draw_light();
  draw_fn();
  draw_xy_plane_if_toggled();
  draw_best_if_toggled();
  draw_particles();
  glutSwapBuffers();
}



void reshape( GLint width, GLint height )
{
  // set new window size for arcball when reshaping
  arcball.set_win_size( width, height );

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective(45, 1.0 * width / height, 1, 1000 );
   
  glViewport(0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void keyboard( GLubyte key, GLint x, GLint y )
{
  if (key == ' ') {
      optimizer.step();
  } else if (key == 'r') {
      optimizer.reset();
  } else if (key == KEY_ESC) {
      exit(0);
  } else if (key == 'h') {
      plane_toggle = !plane_toggle;
  } else if (key == 'g') {
      gbest_toggle = !gbest_toggle;
  } else if (key == 'p') {
      pbest_toggle = !pbest_toggle;
  }
}

void mouse( int button, int state, int x, int y )
{
  // if the left mouse button is pressed
  if( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN )
  {
    bool shift, ctrl, alt = glutGetModifiers();
    // if (shift) {
    // } else {
    {
        // set use of arcball to true 
        arcball_on = true;
        // store current mouse position
        arcball.set_cur( x, y );
        // and begin drag of ball
        arcball.begin_drag();
        // store y values for zooming
        last_y = cur_y = y;
    }
  }
  // else if right button is pressed
  else if( button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN )
  {
    // set use of zooming to true
    zoom_on = true;
    // set use of arcball to false 
    arcball_on = false;
    // and end drag of ball
    arcball.end_drag();
    // store y values for zooming
    last_y = cur_y = y;
  }
  // else set all to false and end drag of ball
  else
  {
    zoom_on = false;
    arcball_on = false;
    arcball.end_drag();
  }
}

void motion( int x, int y )
{
  // id arcball is activated
  if( arcball_on )
  {
    // store current x and y mouse positions
    cur_y = y;
    arcball.set_cur( x, y );
  }

  // if zooming is activated
  if( zoom_on )
  {
    // store current y mouse position for zooming
    cur_y = y;
    
    // if current y value is smaller than last y mouse position
    if( cur_y < last_y )
      // increase zoom rate
      zoomValue += 3.0f;
    else
      // else decrease zoom rate
      zoomValue -= 3.0f;
  }
}

void init_glut() {
  glClearColor(0.1f, 0.1f, 0.15f, 0.0f);
    
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  glEnable(GL_LIGHTING); // enable lighting 
  //glDisable(GL_LIGHTING); // disable lighting 
  glEnable(GL_LIGHT0);
  glEnable(GL_DEPTH_TEST);

  // alpha blending
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
}

int viz_optim(int argc, char** argv)
{
  string update_type = "pso"; // cbo, swarm_grad, pso
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA );
  glutInitWindowSize ( screenWidth, screenHeight );
  glutInitWindowPosition( 0, 0 );

  // Make a somewhat informative title.
  string title = "A Particle swarm finding the minimum of ";
  string func_name = funcName; // convert from char* to string
  title.append(func_name);
  title.append(" using the update type: ");
  title.append(update_type);
  const char * t = title.c_str();
  glutCreateWindow(t);

  init_glut();
  glutReshapeFunc( reshape );
  
  glutIdleFunc( display );
  glutDisplayFunc( display );
  
  glutKeyboardFunc( keyboard );
  glutMouseFunc( mouse );
  glutMotionFunc( motion );

  set_zs();
  set_prtcl_colors(N_GROUPS);

  float lower_bounds[DIMS];
  float upper_bounds[DIMS];
  for (int d=0; d < DIMS; d++) {
	  // float bound = function_bounds[function_name];
	  float bound = 9999999; // 51.2
	  lower_bounds[d] = -bound;
	  upper_bounds[d] = bound;
  }



  // initialize SWARMOPTIMIZER
  float init_range = 0.4;
  float lower_bounds_init_dist[DIMS] = {Xmax[0], Ymax[1]};
  float upper_bounds_init_dist[DIMS] = {Xmax[0] - (Xmax[0]-Xmin[0])*init_range, Ymax[1] - (Ymax[1]-Ymin[1])*init_range};

  // HYPERPARAMETERS
  string initialization = "uniform";
  int merge_time = 200;

  int K;
  float c1, c2, inertia; // CBO does not use inertia weight
  inertia = 0.0;

  // SWARM_GRAD settings for "alpine0"
  if (update_type == "swarm_grad") {
	  inertia = 0.2; // NOTE: ADJUSTABLE PARAMETER
	  c1 = 0.5; // NOTE: ADJUSTABLE PARAMETER
	  c2 = 0.0; // NOTE: ADJUSTABLE PARAMETER
	  K = 3; // swarm_grad reference particles
  } else if (update_type == "cbo") {
	  // CBO settings for "alpine0"
	  c1 = 0.3; // NOTE: ADJUSTABLE PARAMETER
	  c2 = 0.5; // NOTE: ADJUSTABLE PARAMETER
  } else if (update_type == "pso") {
	  // PSO settings for "alpine0"
	  // In most works, c1 = c2 =: c (= 2)
	  inertia = 0.5; // NOTE: ADJUSTABLE PARAMETER
	  c1 = 1.0; // NOTE: ADJUSTABLE PARAMETER
	  c2 = 1.0; // NOTE: ADJUSTABLE PARAMETER
  }

  // initialize the optimizer
  optimizer.init(
      lower_bounds_init_dist,
      upper_bounds_init_dist,
      lower_bounds,
      upper_bounds,
      c1,
      c2,
      inertia,
      initialization,
      update_type,
      "max_steps", // convergence criterion: max_steps, plateau
      2000, // convergence criterion value: number of steps (max/plateau)
      N_GROUPS,
      merge_time,
      objective,
      K
  );

  glutMainLoop();

  return 0;
}

int main( int argc, char** argv )
{
  viz_optim(argc, argv);
}

