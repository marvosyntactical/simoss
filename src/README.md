# Particle Swarm Optimization in C++

See The Makefiles ```Benchfile``` and ```Vizfile``` for compilation instructions.
You must install GLUT before running make.

---
# Usage instructions
## Benchmark
The Benchmark runs a chosen optimizer on an N-dimensional cost function, with the particles restrained to a bounded Box. One run is performed, until some convergence criterion is met.
A few Hyperparameters can be provided as arguments to the script; however you could be interested in looking inside ```benchmark.cpp``` and playing around with them yourself.

```bash

# make Benchmark 
make -f Benchfile

# Run Benchmark with decent hyperparameters:
./Bench pso 2.0 2.0 0.2
# The arguments are (update\_type, c1, c2, w, k)
# --------------  Allowed Values:  -----------------
# Update\_type: pso, cbo, swarm\_grad
# c1: First Hyperparameter of respective Method, Float
# c2: Second Hyperparameter of respective Method, Float
# w: inertia weight, Float
# k: Number of reference particles for swarm\_grad, Float
```

The optimizers are:
 * pso: Particle Swarm Optimization
 * cbo: Consensus Based Optimization (with anisotropic diffusion)
 * swarm\_grad: new, custom optimizer, in dev

## Visualization
The Visualization runs a chosen optimizer on a 2-dimensional cost function, with the particles unrestrained. The Function is plotted near the center of the X-Y plane; you can adjust the grid size and tile width according to the chosen cost function and your hardware.
Look inside ```vis_swarm_2d.cpp``` and play around with the hyperparameters yourself.

```bash

make -f Vizfile
./Vis2D
# -------------- Controls: -------------- 
# Right Mouse Button, Mouse Movement: Zoom I/O
# Left Mouse Button, Mouse Movement: Camera Movement
# Spacebar (can be pressed down): Do an Optimizer Step
# R: Reset the optimizer.
# g: Display an arrow above the global best position+value
# h: Draw the x-y hyperplane

```

### References

The visualization is dependent on code from:

```
@miscellaneous{kroemker,
author={Kroemker, Susanne},
title={Sunlight - Demo of OpenGL},
booktitle={Seminar \textit{Projektive Geometrie mit Anwendungen in der Computergraphik}},
year={2020},
organization={Universität Heidelberg},
email={kroemkerATSIGNiwr.uni-heidelberg.de}
}
```























