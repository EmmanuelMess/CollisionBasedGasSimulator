#define DEBUG true
#if DEBUG
#define PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define PRINT_DEBUG(...) (void) 0
#endif

constant const uint width = 500;
constant const uint height = 500;

constant const uint numberParticles = 20;

constant const float radius = 20;
constant const float dt = 0.5f;

constant const float delta = 1e-6f;
constant const float epsilon = -1e-6f;

struct __attribute__((packed)) Particle {
	float2 position;
	float2 velocity;
};

typedef float Time;

enum CollisionType {
	NONE = 0,
	IGNORE,
	PARTICLE_PARTICLE,
	PARTICLE_WALL_X,
	PARTICLE_WALL_Y
};

struct __attribute__((packed)) Collision {
	enum CollisionType type;
	uint indexB;
};

kernel void calculateIntersectionTime(global const struct Particle* particlesInput,
                                      global Time * const intersectionTimes) {
	const uint i = get_global_id(0);
	const float2 pointA = particlesInput[i].position;
	const float2 velocityA = particlesInput[i].velocity;

	const uint j = get_global_id(1);
	const float2 pointB = particlesInput[j].position;
	const float2 velocityB = particlesInput[j].velocity;

	if (i == j || i < j) {
		return;
	}

	intersectionTimes[i * numberParticles + j] = INFINITY;

	if (hypot(pointA.x - pointB.x, pointA.y - pointB.y) <= 2 * radius) {
		//TODO fix this on point generation
		PRINT_DEBUG("Overlap: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
		return;
	}

	const float a = pow(velocityA.x - velocityB.x, 2) + pow(velocityA.y - velocityB.y, 2);
	const float b = 2 * ((pointA.x - pointB.x) * (velocityA.x - velocityB.x) +(pointA.y - pointB.y) * (velocityA.y - velocityB.y));
	const float c = pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2) - pow(2 * radius, 2);

	const float d = pow(b, 2) - 4 * a * c;

	if (d < 0) {
		PRINT_DEBUG("No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
		return;
	}
	if (b > epsilon) {
		PRINT_DEBUG("Glancing: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
		return;
	}

	const Time t0 = (-b + sqrt(d)) / (2 * a);
	const Time t1 = (-b - sqrt(d)) / (2 * a);

	if (b >= 0) {
		PRINT_DEBUG("Getting farther: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
		return;
	}
	if (t0 < 0 && t1 > 0 && b <= epsilon) {
		PRINT_DEBUG("No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
		return;
	}

	const Time t = t1;
	// The sqrt(delta) prevents issues where the simulation cannot advance at all (e.g. collision at t = 0.0000)
	intersectionTimes[i * numberParticles + j] = max(sqrt(delta), t);

	PRINT_DEBUG("Collision: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f)) at time %f\n", i, pointA.x, pointA.y,
	       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y, t);
}

kernel void collisionTimeParticleWall(const uint i, const float velocity, const float point,
                                      const float wall, local Time * collisionTime) { // TODO const char*?
    const float a = pow(velocity, 2);
    const float b = 2 * (point - wall) * velocity;
    const float c = (point - wall + radius) * (point - wall - radius);

    const float d = pow(b, 2) - 4 * a * c;

    if (d < 0) {
        PRINT_DEBUG("No intersect: %d((%f), (%f)) and ", i, point, velocity);
        *collisionTime = INFINITY;
        return;
    }
    if (b > epsilon) {
        PRINT_DEBUG("Glancing: %d((%f), (%f)) and ", i, point, velocity);
        *collisionTime = INFINITY;
        return;
    }

    const Time t0 = (-b + sqrt(d)) / (2 * a);
    const Time t1 = (-b - sqrt(d)) / (2 * a);

    if (b >= 0) {
        PRINT_DEBUG("Getting farther: %d((%f), (%f)) and ", i, point, velocity);
        *collisionTime = INFINITY;
        return;
    }
    if (t0 < 0 && t1 > 0 && b <= epsilon) {
        PRINT_DEBUG("No intersect: %d((%f), (%f)) and ", i, point, velocity);
        *collisionTime = INFINITY;
        return;
    }

    const Time t = t1;
    // The sqrt(delta) prevents issues where the simulation cannot advance at all (e.g. collision at t = 0.0000)
    *collisionTime = max(sqrt(delta), t);

    PRINT_DEBUG("Collision: %d((%f), (%f)) and ", i, point, velocity);
}

kernel void calculateIntersectionBorderTime(global const struct Particle *positionsInput,
                                            global Time * const intersectionTimes,
                                            global struct Collision * const collidedParticles) {
    const uint i = get_global_id(0);

    local Time t0, t1, t2, t3;// HACK kernel may not have a non-void return
    collisionTimeParticleWall(i, positionsInput[i].velocity.x, positionsInput[i].position.x, 0, &t0); PRINT_DEBUG("%d: Wx = 0 at time %f\n", i, t0); // HACK printf %s must have literal string
    collisionTimeParticleWall(i, positionsInput[i].velocity.x, positionsInput[i].position.x, width, &t1); PRINT_DEBUG("%d: Wx = width at time %f\n", i, t0);
    collisionTimeParticleWall(i, positionsInput[i].velocity.y, positionsInput[i].position.y, 0, &t2); PRINT_DEBUG("%d: Wy = 0 at time %f\n", i, t0);
    collisionTimeParticleWall(i, positionsInput[i].velocity.y, positionsInput[i].position.y, height, &t3); PRINT_DEBUG("%d: Wy = height at time %f\n", i, t0);

    intersectionTimes[i * numberParticles + i] = min(min(t0, t1), min(t2, t3));

    if (min(t0, t1) < min(t2, t3)) {
        collidedParticles[i].type = PARTICLE_WALL_X;
    } else {
        collidedParticles[i].type = PARTICLE_WALL_Y;
    }
}

// TODO do a reduction as recommended by OpenCL
kernel void findMin(global const Time *intersectionTimes, global struct Collision* const collidedParticles,
                    global Time* result) {
    *result = dt;

    bool collision = false; // This is because there could be no collision in the timeframe
    uint indexA = 0;
    uint indexB = 0;

    for (unsigned int i = 0; i < numberParticles; i++) {
        global const Time * rowIntersectionTimes = intersectionTimes + i * numberParticles;

        for (unsigned int j = 0; j < numberParticles; j++) {
            if (i < j) {
                continue;
            }
            PRINT_DEBUG("(%d, %d): %f\n", i, j, intersectionTimes[i * numberParticles + j]);

            const Time intersectionTimeA = intersectionTimes[i * numberParticles + j];

            if (intersectionTimeA < *result) {
                *result = intersectionTimeA;
                indexA = i;
                indexB = j;
                collision = true;
            }
        }
    }

    for (unsigned int i = 0; i < numberParticles; i++) {
        if (collision && indexA == i) {
            continue;
        }

        collidedParticles[i].type = NONE;
    }

    if(!collision) {
        return;
    } else if (indexA == indexB) {
        // Collision type is already set
        return;
    } else {
        collidedParticles[indexA].type = PARTICLE_PARTICLE;
        collidedParticles[indexA].indexB = indexB;

        collidedParticles[indexB].type = IGNORE;
    }
}

kernel void advanceSimulation(global struct Particle * const particlesInput,
                              global struct Particle * const particlesOutput,
                              global const struct Collision * collidingParticles,
                              global const Time* timestepPtr) {

    const Time timestep = *timestepPtr;
    if(timestep == 0) {
        PRINT_DEBUG("Timestep is 0, infinite loop!\n");
        return;
    }

    const uint i = get_global_id(0);

    switch (collidingParticles[i].type) {
        case NONE: {
            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;
            particlesOutput[i].velocity = particlesInput[i].velocity;

            PRINT_DEBUG("%d: No collision!\n", i);
            return;
        }
        case IGNORE: {
            //Another particle is dealing with the collision
            PRINT_DEBUG("%d: Other particle collision!\n", i);
            return;
        }
        case PARTICLE_PARTICLE: {
            const uint indexB = collidingParticles[i].indexB;

            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;
            particlesOutput[indexB].position = particlesInput[indexB].position + timestep * particlesInput[indexB].velocity;

            const float2 positionA = particlesOutput[i].position;
            const float2 positionB = particlesOutput[indexB].position;
            const float2 velocityA = particlesInput[i].velocity;
            const float2 velocityB = particlesInput[indexB].velocity;

            const float2 substract = positionA - positionB;
            const float distanceSquared = pow(substract.x, 2) + pow(substract.y, 2);
            const float product = (velocityA.x - velocityB.x) * (positionA.x - positionB.x)
                                    + (velocityA.y - velocityB.y) * (positionA.y - positionB.y);
            const float2 difference = (product / distanceSquared) * substract;

            // The sqrt(delta) prevents issues where the simulation cannot advance at all (e.g. velocity is small and timestep is small)
            const float2 idealVelocityA = velocityA - difference;
            const float2 velocityCorrectedA = idealVelocityA < sqrt(delta)? 0:idealVelocityA;
            const float2 idealVelocityB = velocityB + difference;
            const float2 velocityCorrectedB = idealVelocityB < sqrt(delta)? 0:idealVelocityB;

#if DEBUG
            const float accumulatedError = (velocityA.x * velocityB.x + velocityA.y * velocityB.y)
                - (velocityCorrectedA.x * velocityCorrectedB.x + velocityCorrectedA.y * velocityCorrectedB.y);
            printf("Collision error: %f", accumulatedError);
#endif

            particlesOutput[i].velocity = velocityCorrectedA;
            particlesOutput[indexB].velocity = velocityCorrectedB;


            PRINT_DEBUG("%d: Particle collision with %d!\n", i, indexB);
            return;
        }
        case PARTICLE_WALL_X: {
            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;
            particlesOutput[i].velocity.x = -particlesInput[i].velocity.x;
            particlesOutput[i].velocity.y = particlesInput[i].velocity.y;
            PRINT_DEBUG("%d: Wall X collision!\n", i);
            return;
        }
        case PARTICLE_WALL_Y: {
            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;
            particlesOutput[i].velocity.x = particlesInput[i].velocity.x;
            particlesOutput[i].velocity.y = -particlesInput[i].velocity.y;
            PRINT_DEBUG("%d: Wall Y collision!\n", i);
            return;
        }
        default:
            PRINT_DEBUG("Wrong particle collision type!\n");
            return;
    }
}
