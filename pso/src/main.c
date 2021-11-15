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

#include "util/arcball.h"
#include "pso.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define KEY_ESC 27


const char* funcName = "schaffersf6";

// global variables for handling the arcball

// screen size
int screenWidth = 1280;
int screenHeight = 720;

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
static const GLfloat back_color[] = {0.0,0.0,1.0,0.5};
static const GLfloat front_color[] = {1.0,1.0,0.1,0.5 };
static const GLfloat front_triangle_color[] = { 0.8,0.8,0.9,0.9 };
static const GLfloat back_triangle_color[] = { 1.0,1.0,1.0,0.9 };
static const GLfloat inner_sphere_color[] = { 0.7,0.7,0.99,1.0 };
static const GLfloat outer_sphere_color[] = { 0.0,1.0,0.0,1.0 };
static const GLfloat semi_transparent_back_color[] = { 0.0,0.0,1.0,0.5};
static const GLfloat semi_transparent_front_color[] = { 1.0,0.0,0.0,0.5 };

/* begin function plot parameter setup */
static const int DIMS = 2;
const GLfloat grid_size = 20.0f;
const GLfloat Xmin[] = {-grid_size,0.0f, 0.0f, 0.0f};
const GLfloat Xmax[] = {grid_size, 0.0f, 0.0f, 0.0f};
const GLfloat Ymin[] = {0.0f, -grid_size, 0.0f, 0.0f};
const GLfloat Ymax[] = {0.0f, grid_size, 0.0f, 0.0f};

static const GLfloat tile_width_x = 0.5;
static const GLfloat tile_width_y = 0.5;

const int num_tiles_x = (Xmax[0] - Xmin[0])/tile_width_x;
const int num_tiles_y = (Ymax[1] - Ymin[1])/tile_width_y;

// GLfloat** zs;
// zs = new GLfloat[num_tiles_x + 1]
// for (int i = 0; i < num_tiles_x + 1; i++) {
//     zs[i] = new GLfloat[num_tiles_y + 1]; // holds z value for each grid vertex
// }
GLfloat zs[num_tiles_x + 1][num_tiles_y + 1]; // holds z value for each grid vertex
/* end function plot parameter setup */


// function prototypes
void draw_fn();
void draw_light();
void draw_coordinate_system( float unit );
void draw_objects();
void display();
void reshape( GLint width, GLint height );
void keyboard( GLubyte key, GLint x, GLint y );
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void init();
int main( int argc, char** argv );
void set_zs();
GLfloat loss_fn (GLfloat X[DIMS]);

GLfloat loss_fn (GLfloat X[DIMS]) {
    GLfloat x = X[0];
    GLfloat y = X[1];
    if (funcName == "rastrigin") {
        GLfloat x_comp = x*x - 10 * cos(2.0 * M_PI * x) + 10;
        GLfloat y_comp = y*y - 10 * cos(2.0 * M_PI * y) + 10;
        return x_comp + y_comp;
    } else if (funcName == "wellblech") {
        return sin(x) - cos(y) - 1.0f;
    } else if (funcName == "schaffersf6") {
        GLfloat denominator = pow((sin(sqrt(x*x + y*y))), 2.0) - 0.5;
        GLfloat numerator = pow(1.0 - 0.001 * (x*x+y*y), 2);
        GLfloat frac = denominator/numerator;
        return - 2.0 - frac;
    }

    return 0.0;
}

/* function definitions */

void set_zs() {
    // get loss values at vertices
    GLfloat x = Xmin[0];
    GLfloat y;
    GLfloat z;
    for (int i = 0; i <= num_tiles_x; i++) {
        y = Ymin[1]; // reset y
        for (int j = 0; j <= num_tiles_y; j++) {
            GLfloat X[2] = {x,y};
            z = loss_fn(X);
            zs[i][j] = z;
            // cout << "at y = " << y << "; x = " << x << endl;
            // cout << "got z = " << z << endl;
            y += tile_width_y;
        }
        x += tile_width_x;
    }
}

void draw_fn() {

    // set triangle color
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, front_triangle_color);
    glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back_triangle_color);

    // now go over vertices again, splitting each square tile into four equilateral triangles facing inwards
    // using the height values we already got
    GLfloat x = Xmin[0];
    GLfloat y;
    for (int i = 0; i < num_tiles_x; i++) {
        y = Ymin[1]; // reset y
        for (int j = 0; j < num_tiles_y; j++) {

            const GLfloat lerp = (zs[i][j] + zs[i+1][j] + zs[i][j+1] + zs[i+1][j+1]) / 4.0f; // midpoint is linear interpolation of four neighboring vertices
            const GLfloat mid[] = {x + (tile_width_x/2.0f), y + (tile_width_y/2.0f), lerp, 0.0f};

            for (int x_vertices = 0; x_vertices < 2; x_vertices++) {
                for (int y_vertices = 0; y_vertices < 2; y_vertices++) {
                    // define outer corner points of triangle
                    const GLfloat v0[] = {x + x_vertices * tile_width_x, y + y_vertices * tile_width_y, zs[i + x_vertices][j + y_vertices]};
                    GLfloat v1[3];
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

                    // set triangle
                    glBegin(GL_TRIANGLES);
                    // glNormal3f(s0[1]*s1[2] - s0[2] * s1[1], s0[2]*s1[0] - s0[0] * s1[2], s0[0] * s1[1] - s0[1] * s1[0]); // normal vector is cross product of triangle sides
                    glVertex3f(v0[0],  v0[1],  v0[2]);
                    glVertex3f(mid[0], mid[1], mid[2]);
                    glVertex3f(v1[0],  v1[1],  v1[2]);
                    glEnd();

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

    static const GLfloat axis_color[]        = {0.5f, 0.5f, 0.5f, 1.0f};

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


void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  gluLookAt(0, 0, 8 + zoomValue, 0, 0, 0, 0, 1, 0);

  // multiply current matrix with arcball matrix
  glMultMatrixf(arcball.get() );

  draw_coordinate_system(1.0f);
  draw_light();
  draw_objects();
  draw_fn();

  glutSwapBuffers();
   
}



void draw_objects()
{
  GLUquadricObj *qobj = gluNewQuadric();

  //glPolygonMode(GL_BACK, GL_FILL);
  //glPolygonMode(GL_FRONT, GL_FILL);
  //glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back_color);
  //glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front_color);

  // blue triangle
  // glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, front_triangle_color);
  // glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back_triangle_color);
  // glBegin(GL_TRIANGLES);
  // glNormal3f(0.0,0.0,-1.0);
  // glVertex3f(3.0,0.0,0.0);
  // glVertex3f(0.0,0.0,0.0);
  // glVertex3f(0.0,3.0,0.0);
  // glEnd();
  // glBegin(GL_TRIANGLES);
  // glNormal3f(0.0,0.0,-1.0);
  // glVertex3f(6.0,0.0,0.0);
  // glVertex3f(3.0,0.0,0.0);
  // glVertex3f(9.0,3.0,0.0);
  // glEnd();



  // inner sphere
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, inner_sphere_color);
  glPushMatrix();
  glTranslatef(1.0,1.0,1.0);
  gluSphere(qobj,0.1,50,50);
  glPopMatrix();


  // Utah teapot available
  //glutSolidTeapot(4.0);
}



void reshape( GLint width, GLint height )
{
  // set new window size for arcball when reshaping
  arcball.set_win_size( width, height );

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective( 45, 1.0 * width / height, 1, 1000 );
   
  glViewport( 0, 0, width, height );

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void keyboard( GLubyte key, GLint x, GLint y )
{
  // just implemented the fast exit functionality
  switch(key)
  {
  case 'q':
  case 'Q':
  case KEY_ESC:
  	exit(0);
	break;
  }
}

void mouse( int button, int state, int x, int y )
{
  // if the left mouse button is pressed
  if( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN )
  {
    bool shift, ctrl, alt = glutGetModifiers();
    if (shift) {


    } else {
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
      zoomValue += 0.7f;
    else
      // else decrease zoom rate
      zoomValue -= 0.7f;
  }
}

void init()
{
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

int main( int argc, char** argv )
{
  glutInit( &argc, argv );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA );
  glutInitWindowSize ( screenWidth, screenHeight );
  glutInitWindowPosition( 0, 0 );
  glutCreateWindow( "A Particle swarm finding the minimum of some function" );

  init();
  glutReshapeFunc( reshape );
  
  glutIdleFunc( display );
  glutDisplayFunc( display );
  
  glutKeyboardFunc( keyboard );
  glutMouseFunc( mouse );
  glutMotionFunc( motion );

  set_zs();
  
  glutMainLoop();
  return 0;
}