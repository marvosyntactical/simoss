#include "pso.h"
#include <random>
using namespace std;

// PSO Implementation

// init function
PSO::PSO(float *lower_bounds, float *upper_bounds, float c1, float c2) {
    infinity = 3.40282e+038;
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

// generator init function
void PSO::init_gen() {
    random_device rd;
    generator = gen(rd());
}


void PSO::step() {
    // before particles are updated; set global best (and local best based on topology)
    // using current positions
    this->update_bests();
    // update position and velocity of each particle
    update_pos();
}

void PSO::init_pos() {
    // randomly initialize particles
    if (INIT == "random") {
        // spawn particles randomly in entire rectangular grid area;
        for (int dim = 0; dim < D; dim++) {
            uniform_real_distribution<> init_dist_d(lower(dim), upper(dim))
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

void PSO::update_bests() {
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
            gbests[i][D] = zi;
        }

    }
}

// advance system one step
void PSO::update_pos() {
    // update position and velocity of each particle
    for (int i = 0; i < N; i++) {
        // retrieve old position and velocity
        v_i = v[i];
        x_i = x[i];
        pbest = pbests[i];

        // calculate new velocity and add it to current pos
        for (int dim = 0; dim < D; dim++) {
            // sample from given distribution
            r1 = DIST(generator);
            r2 = DIST(generator);

            x_i_d = x_i[dim]
            v_i_d = inertia * v_i[dim] + c1 * r1 * (pbest[dim] - x_i_d) + c2 * r2 * (gbest[dim] - x_i_d);

            v[i][dim] = v_i_d;
            x[i][dim] = x_i_d + v_i_d;
        }
    }
}
