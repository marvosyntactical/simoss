#include <random>
#ifndef PSO_H
#define PSO_H

using namespace std;

template <const int N, const int D, typename F, typename DIST, char* INIT>
class PSO // Particle Swarm Optimization
{
	// N: number of particles; D: Dimensionality of search space; 
	// F: (loss) function mapping from R^D -> R; should accept D element float array, return 1 float
	// DIST: distribution object, such as std::uniform_real_distribution(0.0,1.0)
	// INIT: initialization string; see PSO::init_pos
	private:
		// declarations
		float* lower; // D-element array containing lower bounds of hyperrectangular search region
		float* upper; // D-element array containing lower bounds of hyperrectangular search region
		float** x; // position array: N x D
		float** v; // velocity array: N x D
		float* z; // function value array: N
		float c1; // personal best hyperparameter
		float c2; // global best hyperparameter
		float* gbest; // global best array: D + 1 (last element is function value)
		float** pbests; // local best array: N x (D + 1) (last element is function value)
		float inertia;
		mt19937 generator; // mersenne prime based PRNG

		//void init_gen();
		//void init_pos();
		//void update_pos();
		//void update_bests();

		// generator init function
		void init_gen() {
		    random_device rd;
		    generator = gen(rd());
		}

		void init_pos() {
		    // randomly initialize particles
		    if (INIT == "random") {
			// spawn particles randomly in entire rectangular grid area;
			for (int dim = 0; dim < D; dim++) {
			    uniform_real_distribution<float> init_dist_d(lower[dim], upper[dim]);
			    for (int i = 0; i < N; i++) {
				x[i][dim] = init_dist_d(generator);
			    }
			}
			
		    } else if (INIT == "center") {
			// spawn particles randomly on unit sphere around center
			// TODO
			cout << "not implemented: center spawning" << endl;
			exit(1);
		    } else {
			cout << "Allowed values of INIT are 'random', 'center'. Got " << INIT << endl;
			exit(1);
		    }
		    
		}

		void update_bests() {
		    // TODO define different topologies here
		    for (int i = 0; i < N; i++) {
			float x_i[D] = x[i];
			float zi = F(x_i);
			z[i] = zi;
			// TODO make different tie-breaking schemes such as keeping multiple bests
			// update global best if improved
			if (zi < gbest[D]) {
			    for (int dim = 0; dim < D; dim++) {
				gbest[dim] = x_i[dim];
			    }
			    gbest[D] = zi;
			}
			// update personal best if improved
			if (zi < pbests[i][D]) {
			    for (int dim = 0; dim < D; dim++) {
				pbests[i][dim] = x_i[dim];
			    }
			    pbests[i][D] = zi;
			}

		    }
		}

		// advance system one step
		void update_pos() {
		    // update position and velocity of each particle
		    for (int i = 0; i < N; i++) {
			// retrieve old position and velocity
			float v_i[D] = v[i];
			float x_i[D] = x[i];

			// calculate new velocity and add it to current pos
			for (int dim = 0; dim < D; dim++) {
			    // sample from given distribution
			    float r1 = DIST(generator);
			    float r2 = DIST(generator);

			    float x_i_d = x_i[dim];
			    float v_i_d = inertia * v_i[dim] + c1 * r1 * (pbests[i][dim] - x_i_d) + c2 * r2 * (gbest[dim] - x_i_d);

			    v[i][dim] = v_i_d;
			    x[i][dim] = x_i_d + v_i_d;
			}
		    }
		}
	public:
		// void step();
		// void reset();
		// float** get_pos();

		// init function
		PSO(float *lower_bounds, float *upper_bounds, float c1, float c2) {
		    float infinity = 3.40282e+038;
		    init_gen();

		    lower = new float[D];
		    upper = new float[D];

		    x = new float*[N];
		    v = new float*[N];

		    for (int dim = 0; dim < D; dim++) {
			lower[dim] = lower_bounds[dim];
			upper[dim] = upper_bounds[dim];
		    }
		    for (int i = 0; i < N; i++) {
			x[i] = new float*[D];
			v[i] = new float*[D];
			pbests[i][D] = infinity;
		    }
		    gbest[D] = infinity;
		    init_pos();
		}

		void step() {
		    // before particles are updated; set global best (and local best based on topology)
		    // using current positions
		    update_bests();
		    // update position and velocity of each particle
		    update_pos();
		}
};
#endif
