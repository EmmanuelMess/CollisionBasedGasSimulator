#ifndef COLLISIONBASEDGASSIMULATOR_DATATYPES_H
#define COLLISIONBASEDGASSIMULATOR_DATATYPES_H

#include <CL/cl.h>

// All the definitions here must also be in simulator.cl

const cl_uint width = 500;
const cl_uint height = 500;

const cl_uint numberParticles = 20;

const cl_float radius = 20;
const cl_float dt = 0.5f;

struct __attribute__((packed)) Particle {
	cl_float2 position;
	cl_float2 velocity;
};

typedef cl_float Time;

enum CollisionType {
	NONE = 0,
	IGNORE,
	PARTICLE_PARTICLE,
	PARTICLE_WALL_X,
	PARTICLE_WALL_Y
};

struct __attribute__((packed)) Collision {
	enum CollisionType type;
	cl_uint indexB;
};

#endif //COLLISIONBASEDGASSIMULATOR_DATATYPES_H
