/*********************************************
 * Author:
 * Marvin Koss
 * marvinklg at outlook doot de
 *********************************************/

using namespace std;

#include <stdio.h>
#include <stdlib.h>
// std includes
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <map>

#include "optim/swarm.h" // contains swarm optimizer implementations

#define PI 3.14159265

// OBJECTIVE FUNCTION SETUP
const char* function_name = "rastrigin"; // NOTE: ADJUSTABLE PARAMETER
static const int DIMS = 10; // OBJECTIVE PARAMETER

// OPTIMIZER CONSTRUCTION (INITIALIZED LATER)
// SWARMOPTIMIZER variables
static const int N_PARTICLES = 25;
static const int N_GROUPS = 1; // for update_type == "swarm_grad"

SWARMOPTIMIZER<N_PARTICLES, DIMS, float> optimizer;


// contains bound (used as lower and upper in each dim) for each function
std::map<string, float> function_bounds; 

// float function_bounds["rastrigin"] = 5.12;


template <int D, typename numtype>
numtype objective (numtype X[D]) {
    if (function_name == "rastrigin") {
	numtype sum = 0;
	for (int d = 0; d < D; d++) {
		// cout << "Fun: Getting vector value in dim " << d << "... " << endl;

		numtype Xd = X[d];
		sum += Xd*Xd - 10*cos(2*PI*Xd);
	}
        return 10*D+sum;
    } else if (function_name == "wellblech") {
        return 1; // TODO 
    } else if (function_name == "scheffer") {
        return 2; // TODO 
    }
    return 0.0;
}


int main( int argc, char** argv )
{

  // initialize SWARMOPTIMIZER
  float lower_bounds_init_dist[DIMS]; // uniform init dist lower bounds
  float upper_bounds_init_dist[DIMS]; // uniform init dist upper bounds
  for (int d = 0; d < DIMS; d++) {
	  lower_bounds_init_dist[d] = -3;
	  upper_bounds_init_dist[d] = -4;
  }

  float lower_bounds[DIMS]; // uniform init dist lower bounds
  float upper_bounds[DIMS]; // uniform init dist upper bounds
  for (int d=0; d < DIMS; d++) {
	  // float bound = function_bounds[function_name];
	  float bound = 5.12;
	  lower_bounds[d] = -bound;
	  upper_bounds[d] = bound;
  }

  // HYPERPARAMETERS
  string initialization = "uniform"; // uniform: lower[d] < x[i][d] < upper[d] for all i
  int merge_time = 200; // swarm grad: merge groups after this many steps
  string update_type; // take update type as arg // OPTIONS: cbo, swarm_grad, pso
  if (argc > 1) {
	  update_type = argv[1];
	 
  } else {
	  update_type = "swarm_grad";
  }

  float inertia; // intialize always even though CBO does not use inertia weight
  float c1, c2; // correspond to lambda, sigma in case of CBO

  function<float (float*)> obj = objective<DIMS, float>;

  // different hyperparameter settings for different optimizers
  if (update_type == "swarm_grad") {
    // SWARM_GRAD settings for "alpine0"
    // inertia = 0.1; // NOTE: ADJUSTABLE PARAMETER
    c1 = 0.1; // NOTE: ADJUSTABLE PARAMETER
    c2 = 0.1; // NOTE: ADJUSTABLE PARAMETER
    inertia = 0.2;
  } else if (update_type == "cbo") {

    // CBO settings for "alpine0"
    c1 = 0.4; // NOTE: ADJUSTABLE PARAMETER
    c2 = 0.3; // NOTE: ADJUSTABLE PARAMETER

  } else if (update_type == "pso") {
  
    // PSO settings for "alpine0"
    // In most works, c1 = c2 =: c (= 2)
    inertia = 0.2; // NOTE: ADJUSTABLE PARAMETER
    c1 = 0.5; // NOTE: ADJUSTABLE PARAMETER
    c2 = 0.5; // NOTE: ADJUSTABLE PARAMETER
  } else {
    cout << "Not implemented: update_type = " << update_type << endl;
    return 1;
  }

  if (argc > 3) {
	  // string to float
	  c1 = stof(argv[2]);
	  c2 = stof(argv[3]);
  }
  if (argc > 4) {
	  inertia = stof(argv[4]);
  }



  // initialize the optimizer
  // cout << "initializing optimizer ..." << endl;
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
      "plateau",
      10000,
      N_GROUPS,
      merge_time,
      obj
  );


  // output variables
  float* found_optimum;
  float optimum_value;
  int steps_taken;

  // run optimization (bind triple outputs to output variables)
  // cout << "running optimization ..." << endl;
  tie(found_optimum, optimum_value, steps_taken) = optimizer.run();

  /* print out optimum position vector: */
  cout << endl << "=======================================" << endl << endl;
  cout << "Optimizer '" << update_type << "'" << endl;
  cout << "(c1 = " << c1 << ", c2 = " << c2  << ", inertia = " << inertia << ")"  << endl << endl;
  cout << "Found optimum (DIMS=" << DIMS << "):"<< endl;
  cout << "[ ";
  for (int d = 0; d < DIMS-1; d++) {
	  cout << found_optimum[d] << ", " ;
  }
  cout << found_optimum[DIMS-1] << "]" << endl;
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /* print out function value at optimum */
  cout << "Achieving the function value " << optimum_value << endl;
  cout << "After " << steps_taken << " steps. " << endl;
  cout << endl << "=======================================" << endl;

	cout << argc << endl;
  return 0;
}
