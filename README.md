# First-collision timestep gas simulator with OpenCL
This simulator computes all possible intersections, then checks which of those possible intersections is soonest to
occur, and does some checks, the simulator then runs for time for enough time for the collision to occur. Once the 
intersection occurs, the whole process runs again.

The idea is that GPUs allow this to go very fast, by computing a lot of data in parallel (all segments and
intersections), this method of simulation is very precise as intersections are computed analytically, not 
in steps.

## Architecture

The general arch is in [architecture.md](architecture.md).

# Image

<img src="result.webm"/>

# Compilation and running

First compile with:

```bash
cd code
./generate_kernels.sh
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -G Ninja -S ./ -B ./cmake-build-debug
cmake --build ./cmake-build-debug --target CollisionBasedGasSimulator -j 3
```

Then run with:

```bash
./cmake-build-debug/CollisionBasedGasSimulator
```

# Some refrences and thanks

* [Colliding balls](https://garethrees.org/2009/02/17/physics/): An explanation for the basic idea, but without much implementation info.
* [raylib](https://www.raylib.com/): All the drawing, and window management is done with raylib.
* [First-collision timestep gas simulator with CUDA](https://github.com/EmmanuelMess/FirstCollisionTimestepRarefiedGasSimulator): My old implementation of this same algorithm for CUDA
----
<a class="imgpatreon" href="https://www.patreon.com/emmanuelmess" target="_blank">
<img alt="Become a patreon" src="https://user-images.githubusercontent.com/10991116/56376378-07065400-61de-11e9-9583-8ff2148aa41c.png" width=150px></a>
