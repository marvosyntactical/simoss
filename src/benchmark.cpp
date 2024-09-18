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

// #include "vis_swarm_2d.h"
#include "optim/swarm.h" // contains swarm optimizer implementations

#define PI 3.14159265
#define EULER 2.71828

// OBJECTIVE FUNCTION SETUP
const char* function_name = "salomon"; // NOTE: ADJUSTABLE OBJECTIVE PARAMETER
static const int DIMS = 50; // NOTE: ADJUSTABLE OBJECTIVE PARAMETER

static const bool VISUALIZE = false;

// OPTIMIZER CONSTRUCTION (INITIALIZED LATER)
// SWARMOPTIMIZER variables
static const int N_PARTICLES = 100;
static const int N_GROUPS = 1; // for update_type == "swarm_grad"

SWARMOPTIMIZER<N_PARTICLES, DIMS, float> optimizer;


// contains bound (used as lower and upper in each dim) for each function
std::map<string, float> function_bounds; 

// I have to declare this??
// void computeStandardDeviation(float* array, int N, float &stddev);

void computeStandardDeviation(float* array, int N, float &stddev){
    float sum = 0.0f, mean=0.0f, var = 0.0f;

    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    mean = sum/N;

    for (int i = 0; i < N; i++) {
        var += (array[i] - mean) * (array[i] - mean);
    }

    var /= N;
    stddev = sqrt(var);

    delete[] array;
}


template <int D, typename numtype>
numtype norm2 (numtype X[D]) {
	numtype sum = 0.0;
	numtype Xd;
	for (int d = 0; d < D; d++) {
		Xd = X[d];
		sum += Xd*Xd;
	}
	return sqrt(sum);
}

template <int D, typename numtype>
numtype objective (numtype X[D]) {
    if (function_name == "rastrigin") {
	numtype sum = 0;
        numtype A = 10;
	for (int d = 0; d < D; d++) {
		// cout << "Fun: Getting vector value in dim " << d << "... " << endl;

		numtype Xd = X[d];
		sum += Xd*Xd - A*cos(2*PI*Xd) + 10;
	}
        return A*D+sum;
    } else if (function_name == "salomon") {
        numtype sum = 1;
        numtype squares = 0;
	for (int d = 0; d < D; d++) {
            squares += pow(X[d],2);
        }
        squares = sqrt(squares);
        sum += 0.1*squares - cos(2*3.14159*squares);
        return sum;
    } else if (function_name == "xsy4") {
        numtype sines = 0;
        numtype sinesqrts = 0;
        numtype squares = 0;

        for (int d = 0; d < D; d++) {
            sines += pow(sin(X[d]), 2);
            squares += pow(X[d], 2);
            sinesqrts += pow(sin(sqrt(sqrt(pow(X[d],2)))), 2);
        }
        return (sines - exp(-squares)) * exp(-sinesqrts);
    } else if (function_name == "unknown") {
        numtype sum = 0;
        for (int d = 0; d < D; d++) {
            sum += X[d] * X[d];
        }
        return 1 - cos(2 * 3.14159*sqrt(sum)) + 0.1 * sqrt(sum);
    } else if (function_name == "griewank") {
        numtype sum = 1;
        numtype summands = 0;
        numtype factors = 1;
	for (int d = 0; d < D; d++) {
            summands += pow(X[d],2);
            factors *= cos(X[d]/sqrt(d+1));
        }
        summands /= 4000;
        sum += summands + factors;
        return sum;
    } else if (function_name == "wellblech") {
        numtype wobble = 16.0f;
        numtype r = 0;
        for (int d=0; d< D; d++) {
            r += wobble*sin(X[d]*0.1) + 0.004*pow(X[d]+pow(-1,d)*50, 2);
        }
        return r;
    } else if (function_name == "knownMin") {
        numtype r = 0;
        for (int d=0; d< D; d++) {
            r += 0.004*pow(X[d]+pow(-1, d)*50, 2);
        }
        return r;
    } else if (function_name == "ackley") {
	numtype B, C;
	B = 0;
	C = 0;
	numtype cos_sum = 0;
	numtype norm_sum = 0;
	for (int d = 0; d < D; d++) {
		numtype Xd = X[d];
		cos_sum += cos(2*PI*(Xd-B));
		norm_sum += Xd*Xd;
	}
	cos_sum /= D;
	norm_sum = sqrt(norm_sum);
	norm_sum *= -0.2/sqrt(D);

	return -20*exp(norm_sum) - exp(cos_sum)+20+EULER + C;

    } else if (function_name == "scheffer") {
        return 2; // TODO 
    } else if (function_name == "parabola") {
        numtype sum = 0;
        numtype exponent = 2;
        for (int d = 0; d < D; d++) {
            sum += pow(X[d], exponent);
        }
        return sum;
    }
    return 0.0;
}


int main( int argc, char** argv )
{

  // initialize SWARMOPTIMIZER
  float lower_bounds_init_dist[DIMS]; // uniform init dist lower bounds
  float upper_bounds_init_dist[DIMS]; // uniform init dist upper bounds
  for (int d = 0; d < DIMS; d++) {
	  lower_bounds_init_dist[d] = 80;
	  upper_bounds_init_dist[d] = 90;
  }

  float lower_bounds[DIMS]; // uniform init dist lower bounds
  float upper_bounds[DIMS]; // uniform init dist upper bounds
  for (int d=0; d < DIMS; d++) {
	  // float bound = function_bounds[function_name];
	  float bound = 100;
	  lower_bounds[d] = -bound;
	  upper_bounds[d] = bound;
  }

  // HYPERPARAMETERS
  string initialization = "uniform"; // uniform: lower[d] < x[i][d] < upper[d] for all i
  int merge_time = 200; // swarm grad: merge groups after this many steps
  string update_type; // take update type as arg // OPTIONS: cbo, swarm_grad, pso
                      //
  if (argc > 1) {
      update_type = argv[1];
  } else {
      update_type = "pso"; // pso, cbo, cbs, swarm_grad
  }

  float inertia; // initialize always even though CBO does not use inertia weight
  float c1, c2; // correspond to lambda, sigma in case of CBO
  int K = 1; // swarm_grad reference particles
  float temp = 50;
  float beta = .9; // beta2 of adam (inertia is beta1)
  float dt = .1;
  bool do_momentum = true;

  function<float (float*)> obj = objective<DIMS, float>;

  // different hyperparameter settings for different optimizers
  if (update_type == "swarm_grad") {
    // SWARM_GRAD settings for "alpine0"
    c1 = 4.0; // NOTE: ADJUSTABLE PARAMETER
    c2 = 0.8; // NOTE: ADJUSTABLE PARAMETER
    inertia = 0.7; // NOTE: ADJUSTABLE PARAMETER
    beta = 0.9; // NOTE: ADJUSTABLE PARAMETER
    K = 5; // NOTE: ADJUSTABLE PARAMETER

  } else if (update_type == "cbo") {

    // CBO settings for "alpine0"
    c1 = 1.0; // NOTE: ADJUSTABLE PARAMETER
    c2 = 0.1; // NOTE: ADJUSTABLE PARAMETER

  } else if (update_type == "pso") {
  
    // PSO settings for "alpine0"
    // In most works, c1 = c2 =: c (= 2)
    inertia = 0.7; // NOTE: ADJUSTABLE PARAMETER
    c1 = 2.0; // NOTE: ADJUSTABLE PARAMETER
    c2 = 2.0; // NOTE: ADJUSTABLE PARAMETER
  } else if (update_type == "cbs") {
    temp = 30;
    // c2 = 1.0; // optimization mode
    c2 = 1.0/(1.0+temp); // sampling mode
  } else {
    cout << "Not implemented: update_type = " << update_type << endl;
    return 1;
  }

  // CLI config
  if (argc > 3) {
    // string to float
    c1 = stof(argv[2]);
    c2 = stof(argv[3]);
  }
  if (argc > 4) {
    inertia = stof(argv[4]);
  }
  if (argc > 5) {
    K = stof(argv[5]); // swarm_grad number of reference particles
  }
  
  // DO N RUNS AND AVERAGE OVER THEM
  int N_RUNS = 10;

  // MEANS
  float avg_optimum_value;
  int avg_steps_taken;

  // STANDARD DEVIATIONS
  float* optimum_vals = new float[N_RUNS];
  float* steps_takens = new float[N_RUNS];
  float stddev_val = 0.0f;
  float stddev_steps = 0.0f;

  cout << "Benchmarking " << update_type << " on " << function_name << endl;

  for (int i = 0; i < N_RUNS; i++) {
      cout << "Run " << i << endl;

      // initialize the optimizer
      optimizer.init(
          lower_bounds_init_dist,
          upper_bounds_init_dist,
          lower_bounds,
          upper_bounds,
          c1,
          c2,
          inertia,
          beta,
          temp,
          dt,
          initialization,
          update_type,
          "max_steps",
          4000,
          N_GROUPS,
          merge_time,
          obj,
          K,
          do_momentum
      );

      // output variables
      float* found_optimum;
      float optimum_value;
      int steps_taken;

      if (VISUALIZE && DIMS == 2) {
        // TODO NOTE see vis_swarm_2d.cpp
      } else {
        // run optimization (bind triple outputs to output variables)
        // cout << "running optimization ..." << endl;
        tie(found_optimum, optimum_value, steps_taken) = optimizer.run();
      }
      cout << "Optimum " << optimum_value << endl;

      // UPDATE MEANS
      avg_optimum_value += optimum_value/N_RUNS;
      avg_steps_taken += steps_taken/N_RUNS;

      // UPDATE STDS
      optimum_vals[i] = optimum_value;
      steps_takens[i] = steps_taken;

  }

  // compute standard deviations
  computeStandardDeviation(optimum_vals, N_RUNS, stddev_val);
  computeStandardDeviation(steps_takens, N_RUNS, stddev_steps);

  /* print out optimum position vector: */
  cout << endl << "=======================================" << endl << endl;
  cout << "Optimizer '" << update_type << "'" << endl;
  cout << "(c1 = " << c1 << ", c2 = " << c2  << ", inertia = " << inertia << ", K = " << K << ", beta = " << beta << ")"  << endl << endl;
  cout << "Optimized '"<< function_name << "' (DIMS=" << DIMS << ") Over " << N_RUNS << " runs:" << endl << endl;

  // cout << "Found optimum: ";
  // cout << "[ ";
  // for (int d = 0; d < DIMS-1; d++) {
  // 	  cout << found_optimum[d] << ", ";
  // }
  // cout << found_optimum[DIMS-1] << "]" << endl << endl;
  //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /* print out function value at optimum */
  // cout << "With Norm " << norm2<DIMS, float>(found_optimum) << endl;
  cout << "Achieving the average function value " << avg_optimum_value << " (+- " << stddev_val << " )" << endl;
  cout << "After an average number of steps of  " << avg_steps_taken << " (+- " << stddev_steps << " ). " << endl;
  cout << endl << "=======================================" << endl;

  return 0;
}
