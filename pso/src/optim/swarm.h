#include <random>
#include <string>
#include <functional>
#include <tuple>
#include <stdio.h>
#include <math.h>

using namespace std;

// TODO put this class in a separate file again, current problem:
// * make file command includes -c, which avoids linking, so only one file can be provided
// * if -c is omitted, glut raises errors
template <const int N, const int D, typename numtype>
class SWARMOPTIMIZER // Particle Swarm Optimization, CBO, swarmgrad
{
	// N: number of particles; D: Dimensionality of search space; 
	// objective: (loss) function mapping from R^D -> R; should accept D element float array, return 1 float
	// DIST: distribution object, such as uniform_real_distribution(0.0,1.0)
	// initialization: initialization string; see SWARMOPTIMIZER::init_pos
	private:
		// declarations
		function<numtype (numtype*)> objective;
		numtype* upper_bounds_init_dist; // D-element array containing upper_bounds_init_dist bounds of position init dist
		numtype* lower_bounds_init_dist; // D-element array containing lower_bounds_init_dist bounds of position init dist
		numtype* upper_bounds; // D-element array containing upper_bounds_init_dist bounds of hyperrectangular search region
		numtype* lower_bounds; // D-element array containing lower_bounds_init_dist bounds of hyperrectangular search region
		numtype** v; // velocity array: N x D
		numtype c1; // personal best hyperparameter
		numtype c2; // global best hyperparameter
		numtype inertia; // canonical pso omega inertia weight 0 <= w <= 1

		string initialization;
		string update_type;

		mt19937 generator; // mersenne prime based PRNG
		uniform_int_distribution<int> discrete_dist;
		normal_distribution<numtype> continuous_dist;

		string convergence_criterion;
		numtype criterion_value;
		bool converged;
		int plateau_steps;

		// SWARMGRAD:
		int n_groups;
		int group_size;
		int t;
		int merge_time;

		// CBO:
		numtype* softargmax; // soft, smooth weight function value array: N
		numtype* softmax; // X weighted by softargmax
		numtype distance; // distance from X_i to current softmax
		numtype denominator;
		numtype alpha;

		// internal vars to calculate with
		numtype r1;
		numtype r2;
		numtype x_i_d;
		numtype v_attract1;
		numtype v_attract2;
		numtype v_inertial;
		numtype v_personal_best;
		numtype v_global_best;
		numtype v_i_d;
		numtype zi;
		numtype* x_i;
		numtype infinity;

		// generator init function
		void init_gen() {
		    random_device rd;
		    mt19937 generator(rd());
		    this->generator = generator;
		}

		void init_vel() {
		    for (int i = 0; i < N; i++) {
			for (int dim=0; dim  < D; dim++) {
			    v[i][dim] = 0.0;
			}
		    }
		}

		void init_pos() {
		    // randomly initialize particles
		    if (initialization == "uniform") {
			// spawn particles randomly in entire rectangular grid area;
			// cout << "Opt: Initializing particles uniformly ... " << endl;
			for (int dim = 0; dim < D; dim++) {
			    // cout << "Opt: Initializing in dim " <<  dim << "... " << endl;
			    numtype lower_bound_dim = lower_bounds_init_dist[dim];
			    numtype upper_bound_dim = upper_bounds_init_dist[dim];
			    // cout << "Opt: lower: " << lower_bound_dim << "." << endl;
			    // cout << "Opt: upper: " << upper_bound_dim << "." << endl;
			    uniform_real_distribution<numtype> init_dist_d(lower_bound_dim, upper_bound_dim);
			    for (int i = 0; i < N; i++) {
				    // cout << "Opt: Initializing particle " <<  i << "... " << endl;
				x[i][dim] = init_dist_d(generator);
			    }
			}
			for (int i = 0; i < N; i++) {
				// cout << "Opt: Getting function value of particle " <<  i << "... " << endl;
			    z[i] = this->objective(x[i]);
			}
		    } else if (initialization == "center") {
			// spawn particles randomly on unit sphere around center
			// TODO
			// cout << "Not implemented: center spawning" << endl;
			exit(1);
			    } else {
			// cout << "Allowed values of initialization are 'random', 'center'. Got " << initialization << endl;
			exit(1);
		    }
		}

		// update personal bests and global best
		void update_bests() {
		    bool gbest_updated = false;
		    // go over all particles
		    for (int i = 0; i < N; i++) {
			x_i = x[i]; // access particle position vector only once
			zi = this->objective(x_i); // evaluate objective function
			z[i] = zi; // (save function value for visualisation)
			// update personal best if improved
			if (zi < pbests[i][D]) {
			    for (int dim = 0; dim < D; dim++) {
				    pbests[i][dim] = x_i[dim];
			    }
			    pbests[i][D] = zi;
			}
			// update global best if improved
			if (zi < gbest[D]) {
			    for (int dim = 0; dim < D; dim++) {
				gbest[dim] = x_i[dim];
			    }
			    gbest[D] = zi;
			    gbest_updated = true;
			}
		    }
		    if (not gbest_updated) plateau_steps += 1;
		}

		void update_pos() {
		    if (update_type == "pso") {
			update_pos_pso();
		    } else if (update_type == "cbo") {
			update_pos_cbo();
		    } else if (update_type == "swarm_grad") {
			update_pos_swarm_grad();
		    }
		}

		// advance system one step
		void update_pos_pso() {
		    // PSO update
		    // update position and velocity of each particle
		    for (int i = 0; i < N; i++) {
			// calculate new velocity and add it to current pos
			for (int dim = 0; dim < D; dim++) {
			    // sample from uniform distribution
			    // (independently for each dimension!)
			    r1 = this->continuous_dist(generator); // (mersenne prime twister
			    r2 = this->continuous_dist(generator); // mt19937)

			    x_i_d = x[i][dim]; // only access particle position once

			    v_inertial = inertia * v[i][dim];
			    v_personal_best = c1 * r1 * (pbests[i][dim] - x_i_d);
			    v_global_best = c2 * r2 * (gbest[dim] - x_i_d);
			    v_i_d = v_inertial + v_personal_best + v_global_best;

			    // cout << "Opt: Updating value in dim " << dim << " of particle " << i << " ..." << endl;
			    // cout << "v_{t-1}[i][d] " << v[i][dim] << endl;
			    // cout << "v[i][d] " << v_i_d << endl;

			    v[i][dim] = v_i_d; // update velocity
			    // cout << "x[i][dim]." << endl;
			    // cout << "x[i][dim] " << x[i][dim] << endl;
			    x[i][dim] = clamp_pos(x_i_d + v_i_d, dim); // update position
			    // cout << "Opt: Updated value." << endl;
			}
		    }
		}
        
		void update_pos_swarm_grad() {
		    // swarm grad update
		    
		    // upper_bounds_init_dist and lower_bounds_init_dist thresholds on difference (~= gradient clipping)
		    numtype upper_thresh = 5.0;
		    numtype lower_thresh = -0.0; // e.g.lower_bounds_init_dist = (- upper_bounds_init_dist) or = (0.0)
		    numtype mult = 0.1; // update i by this much less if its better
		    for (int i = 0; i < N; i++) {
			// each particle i chooses a comparison particle j
			int j, k; // reference particle j
			j = this->discrete_dist(generator) % N;
			k = this->discrete_dist(generator) % N;

			// if (t < merge_time) {
			//     j = (i + 1) % group_size+int(i/group_size);
			// } else {
			//     if (t == merge_time && i == 0) {
			//         // cout << "MERGING SUBSWARMS at t=" << t << endl;
			//     }
			//     j = (i + 1) % N;
			// }

			numtype diff1 = z[i] - z[j];
			numtype diff2 = z[i] - z[k];
			diff1 = max(min(diff1, upper_thresh), lower_thresh);
			diff2 = max(min(diff2, upper_thresh), lower_thresh);

			// if (difference < 0.0) {
			//     difference *= mult;
			// }

			for (int dim=0; dim < D; dim++) {

			    r1 = this->continuous_dist(generator);
			    r2 = this->continuous_dist(generator);

			    v_inertial = inertia * v[i][dim];

			    // grad ~= (f(x+h)-f(x))/|h| // with h = x[j]-x[i]

			    // go along sampled "gradient" (to reference particle j)
			    v_attract1 = c1 * r1 * diff1 * (x[j][dim] - x[i][dim]);
			    v_attract2 = c1 * r2 * diff2 * (x[k][dim] - x[i][dim]);
			    v_i_d = v_attract1 + v_attract2 + v_inertial;

			    // update
			    v[i][dim] = v_i_d;
			    x[i][dim] = clamp_pos(x[i][dim] + v_i_d, dim);
			}
		    }
		}

		void calc_opt() {
		    // for CBO softmax
		    // 0. clear out cache: softmax (rest is overwritten)
		    for (int d = 0; d < D; d++) {
			softmax[d] = 0.0;
		    }
		    denominator = 0.0; // reset denominator

		    numtype wfi;
		    numtype* x_i = new numtype[D];

		    for (int i = 0;  i < N; i++) {
			wfi = exp(- alpha * z[i]);
			softargmax[i] = wfi;
			denominator += wfi;

		    }
		    for (int i = 0;  i < N; i++) {
			softargmax[i] /= denominator;
		    }
		    for (int i = 0;  i < N; i++) {
			wfi = softargmax[i];
			x_i = x[i];
			for (int d = 0; d < D; d++) {
			    softmax[d] += x_i[d] * wfi;
			}
		    }
		}

		void update_pos_cbo() {
		    // CBO update
		    
		    calc_opt(); // sets this->softmax to approximation of currently optimal particle
		    
		    for (int i = 0; i < N; i++) {
			// calc distance from current optimal
			distance = 0.0;
			for (int dim = 0; dim < D; dim++) {
			    distance += pow(softmax[dim] - x[i][dim], 2);
			}
			distance = sqrt(distance);

			for (int dim = 0; dim < D; dim++) {
			    // sample from uniform distribution
			    // (independently for each dimension!)
			    r2 = this->continuous_dist(generator); // mt19937)

			    x_i_d = x[i][dim]; // only access particle position once

			    numtype dist_d = (softmax[dim] - x_i_d);
			    numtype v_drift = c1 * dist_d; // - lambda * (X-m_t)
			    numtype v_diffuse = c2 * r2 * distance;
			    v_i_d = v_drift + v_diffuse;

			    v[i][dim] = v_i_d; // update velocity
			    x[i][dim] = clamp_pos(x[i][dim] + v_i_d, dim); // update position
			}
		    }
		}

		numtype clamp_pos(numtype pos, int dim) {
		    // cout << "clamping to dim" << dim << endl;
			return max(min(pos, upper_bounds[dim]), lower_bounds[dim]);
		}

	public:

		// make these fields accessible for visualization (drawing an arrow above)
		numtype* gbest; // global best array: D + 1 (last element is function value)
		numtype** pbests; // local best array: N x (D + 1) (last element is function value)

		// make these fields accessible vor visualization (drawing particles)
		numtype** x; // position array: N x D
		numtype* z; // function value array: N

		// constructor
		SWARMOPTIMIZER(){}

		// initialize after constructing
		void init(
			numtype *lower_bounds_init_dist, // positions
			numtype *upper_bounds_init_dist, // positions
			numtype *lower_bounds, // positions
			numtype *upper_bounds, // positions
			numtype c1, // HYPERPARAM
			numtype c2, // HYPERPARAM
			numtype inertia, // HYPERPARAM
			string initialization,
			string update_type, // ALGO: pso, cbo, swarm_grad?
			string convergence_criterion,
			numtype criterion_value,
			int n_groups,
			int merge_time,
			const function<numtype (numtype*)>& objective
		    ) {

		    uniform_int_distribution<int> discrete_dist(0, N-1);
		    normal_distribution<numtype> continuous_dist(.0, 1.);

		    this->initialization = initialization;
		    this->inertia = inertia;
		    this->c1 = c1;
		    this->c2 = c2;

		    // optimizer type: pso, cbo, swarm_grad?
		    this->update_type = update_type;

		    // swarm grad
		    this->n_groups = n_groups;
		    this->group_size = int(N/n_groups);
		    this->merge_time = merge_time;

		    this->objective = objective;
		    this->convergence_criterion = convergence_criterion;
		    this->criterion_value = criterion_value;

		    alpha = 1.0; // for cbo
		    infinity = 3.40282e+038;
		    init_gen();

		    z = new numtype[N];
			    
		    this->lower_bounds = new numtype[D];
		    this->upper_bounds = new numtype[D];
		    this->lower_bounds_init_dist = new numtype[D];
		    this->upper_bounds_init_dist = new numtype[D];
		    softmax = new numtype[D];

		    for (int dim = 0; dim < D; dim++) {
			this->lower_bounds_init_dist[dim] = lower_bounds_init_dist[dim];
			this->upper_bounds_init_dist[dim] = upper_bounds_init_dist[dim];
			this->lower_bounds[dim] = lower_bounds[dim];
			this->upper_bounds[dim] = upper_bounds[dim];
			softmax[dim] = 0;
		    }

		    x = new numtype*[N];
		    v = new numtype*[N];
		    softargmax = new numtype[N];
		    
		    pbests = new numtype*[N];

		    for (int i = 0; i < N; i++) {
			x[i] = new numtype[D];
			v[i] = new numtype[D];
			pbests[i] = new numtype[D+1];
		    }
		    gbest = new numtype[D+1];

		    reset();
		}

		void reset() {

		    // cout << "Opt: Resetting memory ... " << endl;
		    // reset memory
		    gbest[D] = infinity;
		    for (int i = 0; i < N; i++) {
			pbests[i][D] = infinity;
		    }

		    // cout << "Opt: Resetting particles ... " << endl;
		    // initialize particle distribution
		    init_pos();
		    init_vel();

		    // reset optimizer
		    // cout << "Opt: Resetting control vars ... " << endl;
		    converged = false;
		    plateau_steps = 0;
		    t = 0;
		}

		void step() {
		    // call individual updates manually, e.g.
		    // in visualization loop

		    // before particles are updated:
		    // set global best (and local best based on topology)
		    // using current positions
		    // cout << "Opt: Updating bests ..." << endl;
		    update_bests();
		    // update position and velocity of each particle
		    // cout << "Opt: Updating positions ..." << endl;
		    update_pos();
		    t += 1;
		}

		bool is_converged() {
		    if (convergence_criterion == "max_steps") {
			converged = t >= criterion_value;
		    } else if (convergence_criterion == "plateau") {
			// plateaued for n steps?
			converged = plateau_steps > criterion_value;
		    } else if (convergence_criterion == "value") {
			converged = gbest[D] <= criterion_value;
		    } else {
		        cout << "Not implemented: convergence criterion = " << convergence_criterion;
		    };
		    return converged;
		}

		tuple<numtype*, numtype, int> run() {
		    // reset, then do steps until convergence criterion met
		    // and return gbest/ optimal found position
		    // cout << "Resetting optimizer ... " << endl;
		    reset();
		    // cout << "Reset optimizer." << endl;
		    while (not is_converged()) {
			    // cout << "Opt: performing step " << t << " ... " << endl;
			step();
			// TODO print out some validation every 100 steps or so
		    }

		    // gbest is D+1 dimensional (gbest[-1] = objective(gbest[:-1]))
		    numtype* best_position = new numtype[D];
		    for (int d = 0; d < D; d++) {
			   best_position[d] = gbest[d];
		    }
		    // cout << "Returning results ..." << endl;
		    return make_tuple(
			   best_position,
			   gbest[D],
			   t
		    ); 
		}
};

