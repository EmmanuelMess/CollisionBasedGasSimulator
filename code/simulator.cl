// This file only contains the simulator code

// THIS FILE IS GENERATED, DO NOT EDIT

const char simulatorKernels[] = "#define DEBUG false\n"
"#if DEBUG\n"
"#define PRINT_DEBUG printf\n"
"#else\n"
"#define PRINT_DEBUG //\n"
"#endif\n"
"\n"
"constant const uint width = 500;\n"
"constant const uint height = 500;\n"
"\n"
"constant const uint numberParticles = 20;\n"
"\n"
"constant const float radius = 20;\n"
"constant const float dt = 0.5f;\n"
"\n"
"constant const float delta = 1e-6f;\n"
"constant const float epsilon = -1e-6f;\n"
"\n"
"struct __attribute__((packed)) Particle {\n"
"	float2 position;\n"
"	float2 velocity;\n"
"};\n"
"\n"
"typedef float Time;\n"
"\n"
"enum CollisionType {\n"
"	NONE = 0,\n"
"	IGNORE,\n"
"	PARTICLE_PARTICLE,\n"
"	PARTICLE_WALL_X,\n"
"	PARTICLE_WALL_Y\n"
"};\n"
"\n"
"struct __attribute__((packed)) Collision {\n"
"	enum CollisionType type;\n"
"	uint indexB;\n"
"};\n"
"\n"
"kernel void calculateIntersectionTime(global const struct Particle* particlesInput,\n"
"                                      global Time * const intersectionTimes) {\n"
"	const uint i = get_global_id(0);\n"
"	const float2 pointA = particlesInput[i].position;\n"
"	const float2 velocityA = particlesInput[i].velocity;\n"
"\n"
"	const uint j = get_global_id(1);\n"
"	const float2 pointB = particlesInput[j].position;\n"
"	const float2 velocityB = particlesInput[j].velocity;\n"
"\n"
"	if (i == j || i < j) {\n"
"		return;\n"
"	}\n"
"\n"
"	intersectionTimes[i * numberParticles + j] = INFINITY;\n"
"\n"
"	if (hypot(pointA.x - pointB.x, pointA.y - pointB.y) <= 2 * radius) {\n"
"		//TODO fix this on point generation\n"
"		PRINT_DEBUG(\"Overlap: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\\n\", i, pointA.x, pointA.y,\n"
"		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);\n"
"		return;\n"
"	}\n"
"\n"
"	const float a = pow(velocityA.x - velocityB.x, 2) + pow(velocityA.y - velocityB.y, 2);\n"
"	const float b = 2 * ((pointA.x - pointB.x) * (velocityA.x - velocityB.x) +(pointA.y - pointB.y) * (velocityA.y - velocityB.y));\n"
"	const float c = pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2) - pow(2 * radius, 2);\n"
"\n"
"	const float d = pow(b, 2) - 4 * a * c;\n"
"\n"
"	if (d < 0) {\n"
"		PRINT_DEBUG(\"No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\\n\", i, pointA.x, pointA.y,\n"
"		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);\n"
"		return;\n"
"	}\n"
"	if (b > epsilon) {\n"
"		PRINT_DEBUG(\"Glancing: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\\n\", i, pointA.x, pointA.y,\n"
"		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);\n"
"		return;\n"
"	}\n"
"\n"
"	const Time t0 = (-b + sqrt(d)) / (2 * a);\n"
"	const Time t1 = (-b - sqrt(d)) / (2 * a);\n"
"\n"
"	if (b >= 0) {\n"
"		PRINT_DEBUG(\"Getting farther: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\\n\", i, pointA.x, pointA.y,\n"
"		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);\n"
"		return;\n"
"	}\n"
"	if (t0 < 0 && t1 > 0 && b <= epsilon) {\n"
"		PRINT_DEBUG(\"No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\\n\", i, pointA.x, pointA.y,\n"
"		       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);\n"
"		return;\n"
"	}\n"
"\n"
"	const Time t = t1;\n"
"	// The delta prevents issues where the simulation cannot advance at all (e.g. collision at t = 0.0000)\n"
"	intersectionTimes[i * numberParticles + j] = max(delta, t);\n"
"\n"
"	PRINT_DEBUG(\"Collision: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f)) at time %f\\n\", i, pointA.x, pointA.y,\n"
"	       velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y, t);\n"
"}\n"
"\n"
"kernel void collisionTimeParticleWall(const uint i, const float velocity, const float point,\n"
"                                      const float wall, local Time * collisionTime) { // TODO const char*?\n"
"    const float a = pow(velocity, 2);\n"
"    const float b = 2 * (point - wall) * velocity;\n"
"    const float c = (point - wall + radius) * (point - wall - radius);\n"
"\n"
"    const float d = pow(b, 2) - 4 * a * c;\n"
"\n"
"    if (d < 0) {\n"
"        PRINT_DEBUG(\"No intersect: %d((%f), (%f)) and \", i, point, velocity);\n"
"        *collisionTime = INFINITY;\n"
"        return;\n"
"    }\n"
"    if (b > epsilon) {\n"
"        PRINT_DEBUG(\"Glancing: %d((%f), (%f)) and \", i, point, velocity);\n"
"        *collisionTime = INFINITY;\n"
"        return;\n"
"    }\n"
"\n"
"    const Time t0 = (-b + sqrt(d)) / (2 * a);\n"
"    const Time t1 = (-b - sqrt(d)) / (2 * a);\n"
"\n"
"    if (b >= 0) {\n"
"        PRINT_DEBUG(\"Getting farther: %d((%f), (%f)) and \", i, point, velocity);\n"
"        *collisionTime = INFINITY;\n"
"        return;\n"
"    }\n"
"    if (t0 < 0 && t1 > 0 && b <= epsilon) {\n"
"        PRINT_DEBUG(\"No intersect: %d((%f), (%f)) and \", i, point, velocity);\n"
"        *collisionTime = INFINITY;\n"
"        return;\n"
"    }\n"
"\n"
"    const Time t = t1;\n"
"    // The delta prevents issues where the simulation cannot advance at all (e.g. collision at t = 0.0000)\n"
"    *collisionTime = max(delta, t);\n"
"\n"
"    PRINT_DEBUG(\"Collision: %d((%f), (%f)) and \", i, point, velocity);\n"
"}\n"
"\n"
"kernel void calculateIntersectionBorderTime(global const struct Particle *positionsInput,\n"
"                                            global Time * const intersectionTimes,\n"
"                                            global struct Collision * const collidedParticles) {\n"
"    const uint i = get_global_id(0);\n"
"\n"
"    local Time t0, t1, t2, t3;// HACK kernel may not have a non-void return\n"
"    collisionTimeParticleWall(i, positionsInput[i].velocity.x, positionsInput[i].position.x, 0, &t0); PRINT_DEBUG(\"%d: Wx = 0 at time %f\\n\", i, t0); // HACK printf %s must have literal string\n"
"    collisionTimeParticleWall(i, positionsInput[i].velocity.x, positionsInput[i].position.x, width, &t1); PRINT_DEBUG(\"%d: Wx = width at time %f\\n\", i, t0);\n"
"    collisionTimeParticleWall(i, positionsInput[i].velocity.y, positionsInput[i].position.y, 0, &t2); PRINT_DEBUG(\"%d: Wy = 0 at time %f\\n\", i, t0);\n"
"    collisionTimeParticleWall(i, positionsInput[i].velocity.y, positionsInput[i].position.y, height, &t3); PRINT_DEBUG(\"%d: Wy = height at time %f\\n\", i, t0);\n"
"\n"
"    intersectionTimes[i * numberParticles + i] = min(min(t0, t1), min(t2, t3));\n"
"\n"
"    if (min(t0, t1) < min(t2, t3)) {\n"
"        collidedParticles[i].type = PARTICLE_WALL_X;\n"
"    } else {\n"
"        collidedParticles[i].type = PARTICLE_WALL_Y;\n"
"    }\n"
"}\n"
"\n"
"// TODO do a reduction as recommended by OpenCL\n"
"kernel void findMin(global const Time *intersectionTimes, global struct Collision* const collidedParticles,\n"
"                    global Time* result) {\n"
"    *result = dt;\n"
"\n"
"    bool collision = false; // This is because there could be no collision in the timeframe\n"
"    uint indexA = 0;\n"
"    uint indexB = 0;\n"
"\n"
"    for (unsigned int i = 0; i < numberParticles; i++) {\n"
"        global const Time * rowIntersectionTimes = intersectionTimes + i * numberParticles;\n"
"\n"
"        for (unsigned int j = 0; j < numberParticles; j++) {\n"
"            if (i < j) {\n"
"                continue;\n"
"            }\n"
"            PRINT_DEBUG(\"(%d, %d): %f\\n\", i, j, intersectionTimes[i * numberParticles + j]);\n"
"\n"
"            const Time intersectionTimeA = intersectionTimes[i * numberParticles + j];\n"
"\n"
"            if (intersectionTimeA < *result) {\n"
"                *result = intersectionTimeA;\n"
"                indexA = i;\n"
"                indexB = j;\n"
"                collision = true;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    for (unsigned int i = 0; i < numberParticles; i++) {\n"
"        if (collision && indexA == i) {\n"
"            continue;\n"
"        }\n"
"\n"
"        collidedParticles[i].type = NONE;\n"
"    }\n"
"\n"
"    if(!collision) {\n"
"        return;\n"
"    } else if (indexA == indexB) {\n"
"        // Collision type is already set\n"
"        return;\n"
"    } else {\n"
"        collidedParticles[indexA].type = PARTICLE_PARTICLE;\n"
"        collidedParticles[indexA].indexB = indexB;\n"
"\n"
"        collidedParticles[indexB].type = IGNORE;\n"
"    }\n"
"}\n"
"\n"
"kernel void advanceSimulation(global struct Particle * const particlesInput,\n"
"                              global struct Particle * const particlesOutput,\n"
"                              global const struct Collision * collidingParticles,\n"
"                              global const Time* timestepPtr) {\n"
"\n"
"    const Time timestep = *timestepPtr;\n"
"    if(timestep == 0) {\n"
"        PRINT_DEBUG(\"Timestep is 0, infinite loop!\\n\");\n"
"        return;\n"
"    }\n"
"\n"
"    const uint i = get_global_id(0);\n"
"\n"
"    switch (collidingParticles[i].type) {\n"
"        case NONE: {\n"
"            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;\n"
"            particlesOutput[i].velocity = particlesInput[i].velocity;\n"
"\n"
"            PRINT_DEBUG(\"%d: No collision!\", i);\n"
"            return;\n"
"        }\n"
"        case IGNORE: {\n"
"            //Another particle is dealing with the collision\n"
"            PRINT_DEBUG(\"%d: Other particle collision!\", i);\n"
"            return;\n"
"        }\n"
"        case PARTICLE_PARTICLE: {\n"
"            const uint indexB = collidingParticles[i].indexB;\n"
"\n"
"            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;\n"
"            particlesOutput[indexB].position = particlesInput[indexB].position + timestep * particlesInput[indexB].velocity;\n"
"\n"
"            const float lengthA = hypot(particlesInput[i].velocity.x, particlesInput[i].velocity.y);\n"
"            const float2 normalizedVelocityA = particlesInput[i].velocity / lengthA;\n"
"\n"
"            const float lengthB = hypot(particlesInput[indexB].velocity.x, particlesInput[indexB].velocity.y);\n"
"            const float2 normalizedVelocityB = particlesInput[indexB].velocity / lengthB;\n"
"\n"
"            particlesOutput[i].velocity = normalizedVelocityB * lengthA;\n"
"            particlesOutput[indexB].velocity = normalizedVelocityA * lengthB;\n"
"            PRINT_DEBUG(\"%d: Particle collision with %d!\", i, indexB);\n"
"            return;\n"
"        }\n"
"        case PARTICLE_WALL_X: {\n"
"            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;\n"
"            particlesOutput[i].velocity.x = -particlesInput[i].velocity.x;\n"
"            particlesOutput[i].velocity.y = particlesInput[i].velocity.y;\n"
"            PRINT_DEBUG(\"%d: Wall X collision!\", i);\n"
"            return;\n"
"        }\n"
"        case PARTICLE_WALL_Y: {\n"
"            particlesOutput[i].position = particlesInput[i].position + timestep * particlesInput[i].velocity;\n"
"            particlesOutput[i].velocity.x = particlesInput[i].velocity.x;\n"
"            particlesOutput[i].velocity.y = -particlesInput[i].velocity.y;\n"
"            PRINT_DEBUG(\"%d: Wall Y collision!\", i);\n"
"            return;\n"
"        }\n"
"        default:\n"
"            PRINT_DEBUG(\"Wrong particle collision type!\\n\");\n"
"            return;\n"
"    }\n"
"}\n"
"";