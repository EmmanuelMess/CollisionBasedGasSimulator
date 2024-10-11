#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <raylib.h>

#define nullptr NULL

#include "datatypes.h"
#include "simulator.cl"

cl_float2 generatePosition() {
	const cl_float x = radius + fmodf((cl_float) rand(), (cl_float) width - radius * 2);
	const cl_float y = radius + fmodf((cl_float) rand(), (cl_float) height - radius * 2);

	return (cl_float2) { .x = x, .y = y };
}

cl_float2 generateVelocity() {
	const cl_float length = 20;

	const cl_float x = 0.1f + fmodf((cl_float) rand(), 10.0f);
	const cl_float y = 0.1f + fmodf((cl_float) rand(), 10.0f);

	const cl_float randomLength = hypotf(x, y);

	return (cl_float2) { .x = x / randomLength * length, .y = y / randomLength * length };
}

struct ClState {
	cl_platform_id platform;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;

	bool success;
};

struct ClState initClState(bool gpu) {
	struct ClState clState;

	{ // Get available platforms
		cl_uint numberOfPlatforms;
		const cl_int err = clGetPlatformIDs(1, &clState.platform, &numberOfPlatforms);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create get a platform! %d\n", err);
			clState.success = false;
			return clState;
		}
	}

	{ // Connect to a compute device
		const cl_int err = clGetDeviceIDs(clState.platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &clState.device_id, nullptr);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create a device group! %d\n", err);
			clState.success = false;
			return clState;
		}
	}

	{ // Create a compute context
		cl_int err;
		clState.context = clCreateContext(nullptr, 1, &clState.device_id, nullptr, NULL, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create a compute context! %d\n", err);
			clState.success = false;
			return clState;
		}
	}

	{ // Create a command commands
		cl_int err;
		clState.commands = clCreateCommandQueueWithProperties(clState.context, clState.device_id, nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create a command commands! %d\n", err);
			clState.success = false;
			return clState;
		}
	}

	{ // Create the compute program from the source buffer
		cl_int err;

		char* sources[] = { (char*) simulatorKernels, nullptr};
		clState.program = clCreateProgramWithSource(clState.context, 1, (const char **) sources, nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create compute program! %d\n", err);
			clState.success = false;
			return clState;
		}
	}

	{ // Build the program executable
		cl_int err = clBuildProgram(clState.program, 0, nullptr, nullptr, nullptr, NULL);
		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[100*1024];

			printf("Error: Failed to build program executable! %d\n", err);
			clGetProgramBuildInfo(clState.program, clState.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			clState.success = false;
			return clState;
		}
	}

	clState.success = true;
	return clState;
}

void releaseClState(struct ClState clState) {
	clReleaseProgram(clState.program);
	clReleaseCommandQueue(clState.commands);
	clReleaseContext(clState.context);
}

struct ClSimulationKernel {
	cl_kernel calculateIntersectionTimeKernel;
	cl_kernel calculateIntersectionBorderTimeKernel;
	cl_kernel findMinKernel;
	cl_kernel advanceSimulationKernel;

	cl_mem particlesInput;
	cl_mem particlesOutput;
	cl_mem intersectionTimes;
	cl_mem collidedParticles;
	cl_mem minimumTime;

	cl_event writeParticlePositionsEvent;
	cl_event writeParticleVelocitiesEvent;
	cl_event writeIntersectionTimesEvent;

	bool success;
};

struct ClSimulationKernel initSimulationKernel(struct ClState clState, struct Particle* particlesInput,
                                               struct Particle* particlesOutput, Time* intersectionTimes) {
	struct ClSimulationKernel clSimulationKernel;

	if(!clState.success) {
		clSimulationKernel.success = false;
		return clSimulationKernel;
	}

	{ // Create the compute kernel in the program we wish to run
		cl_int err;
		clSimulationKernel.calculateIntersectionTimeKernel = clCreateKernel(clState.program,
																			"calculateIntersectionTime", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the compute kernel in the program we wish to run
		cl_int err;
		clSimulationKernel.calculateIntersectionBorderTimeKernel = clCreateKernel(clState.program,
		                                                                    "calculateIntersectionBorderTime", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the compute kernel in the program we wish to run
		cl_int err;
		clSimulationKernel.findMinKernel = clCreateKernel(clState.program, "findMin", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the compute kernel in the program we wish to run
		cl_int err;
		clSimulationKernel.advanceSimulationKernel = clCreateKernel(clState.program, "advanceSimulation", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the input array in device memory for our calculation
		cl_int err;
		clSimulationKernel.particlesInput = clCreateBuffer(clState.context, CL_MEM_READ_WRITE,
		                                                      sizeof(typeof(*particlesInput)) * numberParticles,
		                                                   nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the output array in device memory for our calculation
		cl_int err;
		clSimulationKernel.particlesOutput = clCreateBuffer(clState.context, CL_MEM_READ_WRITE,
		                                                   sizeof(typeof(*particlesOutput)) * numberParticles,
		                                                   nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the output array in device memory for our calculation
		cl_int err;
		clSimulationKernel.intersectionTimes = clCreateBuffer(clState.context, CL_MEM_READ_WRITE,
		                                                      sizeof(typeof(*intersectionTimes)) * (numberParticles * numberParticles),
															  nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the output array in device memory for our calculation
		cl_int err;
		clSimulationKernel.collidedParticles = clCreateBuffer(clState.context, CL_MEM_READ_WRITE,
		                                                      sizeof(struct Collision) * numberParticles,
		                                                      nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	{ // Create the output array in device memory for our calculation
		cl_int err;
		clSimulationKernel.minimumTime = clCreateBuffer(clState.context, CL_MEM_READ_WRITE, sizeof(Time), nullptr, &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to allocate device memory! %d\n", err);
			clSimulationKernel.success = false;
			return clSimulationKernel;
		}
	}

	clSimulationKernel.success = true;
	return clSimulationKernel;
}

void releaseClSimulationKernel(struct ClSimulationKernel clSimulationKernel) {
	clReleaseKernel(clSimulationKernel.calculateIntersectionTimeKernel);
	clReleaseKernel(clSimulationKernel.calculateIntersectionBorderTimeKernel);
	clReleaseKernel(clSimulationKernel.findMinKernel);
	clReleaseKernel(clSimulationKernel.advanceSimulationKernel);

	clReleaseMemObject(clSimulationKernel.particlesInput);
	clReleaseMemObject(clSimulationKernel.particlesOutput);
	clReleaseMemObject(clSimulationKernel.intersectionTimes);
	clReleaseMemObject(clSimulationKernel.collidedParticles);
	clReleaseMemObject(clSimulationKernel.minimumTime);

	clReleaseEvent(clSimulationKernel.writeParticlePositionsEvent);
	clReleaseEvent(clSimulationKernel.writeParticleVelocitiesEvent);
	clReleaseEvent(clSimulationKernel.writeIntersectionTimesEvent);
}

int callSimulation(struct ClState clState, struct ClSimulationKernel clSimulationKernel) {
	{ // calculateIntersectionTime(particlesInput, intersectionTimes);
		{ // Set the arguments to our compute kernel
			cl_int err = clSetKernelArg(clSimulationKernel.calculateIntersectionTimeKernel, 0,
			                            sizeof(typeof(clSimulationKernel.particlesInput)),
			                            &clSimulationKernel.particlesInput);
			err |= clSetKernelArg(clSimulationKernel.calculateIntersectionTimeKernel, 1,
			                      sizeof(typeof(clSimulationKernel.intersectionTimes)),
			                      &clSimulationKernel.intersectionTimes);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set kernel arguments! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		size_t local;
		{ // Get the maximum work group size for executing the kernel on the device
			cl_int err = clGetKernelWorkGroupInfo(clSimulationKernel.calculateIntersectionTimeKernel, clState.device_id,
			                                      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to retrieve kernel work group info! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Execute the kernel over the entire range of our 1d input data set using the maximum number of
			// work group items for this device
			size_t global[2] = { numberParticles, numberParticles };// TODO fix global group size
			local = 1;// TODO fix workgroup size
			cl_int err = clEnqueueNDRangeKernel(clState.commands, clSimulationKernel.calculateIntersectionTimeKernel, 2,
			                                    nullptr, global, &local, 0, nullptr, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Wait for the command commands to get serviced before reading back results
			clFinish(clState.commands); // TODO add to dependency list on the next read
		}
	}
	{ // calculateIntersectionBorderTime(initialPositions, intersectionTimes, collidedParticles);
		{ // Set the arguments to our compute kernel
			cl_int err = clSetKernelArg(clSimulationKernel.calculateIntersectionBorderTimeKernel, 0,
			                            sizeof(typeof(clSimulationKernel.particlesInput)),
			                            &clSimulationKernel.particlesInput);
			err |= clSetKernelArg(clSimulationKernel.calculateIntersectionBorderTimeKernel, 1,
			                      sizeof(typeof(clSimulationKernel.intersectionTimes)),
			                      &clSimulationKernel.intersectionTimes);
			err |= clSetKernelArg(clSimulationKernel.calculateIntersectionBorderTimeKernel, 2,
			                      sizeof(typeof(clSimulationKernel.collidedParticles)),
			                      &clSimulationKernel.collidedParticles);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set kernel arguments! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		size_t local;
		{ // Get the maximum work group size for executing the kernel on the device
			cl_int err = clGetKernelWorkGroupInfo(clSimulationKernel.calculateIntersectionBorderTimeKernel, clState.device_id,
			                                      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to retrieve kernel work group info! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Execute the kernel over the entire range of our 1d input data set using the maximum number of
			// work group items for this device
			size_t global = numberParticles;// TODO fix global group size
			local = 1;// TODO fix workgroup size
			cl_int err = clEnqueueNDRangeKernel(clState.commands, clSimulationKernel.calculateIntersectionBorderTimeKernel, 1,
			                                    nullptr, &global, &local, 0, nullptr, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Wait for the command commands to get serviced before reading back results
			clFinish(clState.commands); // TODO add to dependency list on the next read
		}
	}
	{ // findMin(intersectionTimes, collidedParticles, minimumTime);
		{ // Set the arguments to our compute kernel
			cl_int err = clSetKernelArg(clSimulationKernel.findMinKernel, 0,
			                      sizeof(typeof(clSimulationKernel.intersectionTimes)),
			                      &clSimulationKernel.intersectionTimes);
			err |= clSetKernelArg(clSimulationKernel.findMinKernel, 1,
			                      sizeof(typeof(clSimulationKernel.collidedParticles)),
			                      &clSimulationKernel.collidedParticles);
			err |= clSetKernelArg(clSimulationKernel.findMinKernel, 2,
			                      sizeof(typeof(clSimulationKernel.minimumTime)),
			                      &clSimulationKernel.minimumTime);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set kernel arguments! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		size_t local;
		{ // Get the maximum work group size for executing the kernel on the device
			cl_int err = clGetKernelWorkGroupInfo(clSimulationKernel.findMinKernel, clState.device_id,
			                                      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to retrieve kernel work group info! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Execute the kernel over the entire range of our 1d input data set using the maximum number of
			// work group items for this device
			size_t global = 1;// TODO fix global group size
			local = 1;// TODO fix workgroup size
			cl_int err = clEnqueueNDRangeKernel(clState.commands, clSimulationKernel.findMinKernel, 1,
			                                    nullptr, &global, &local, 0, nullptr, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Wait for the command commands to get serviced before reading back results
			clFinish(clState.commands); // TODO add to dependency list on the next read
		}
	}
	{ // advanceSimulation(particlesInput, particlesOutput, collidedParticles, minimumTime);
		{ // Set the arguments to our compute kernel
			cl_int err = clSetKernelArg(clSimulationKernel.advanceSimulationKernel, 0,
			                            sizeof(typeof(clSimulationKernel.particlesInput)),
			                            &clSimulationKernel.particlesInput);
			err |= clSetKernelArg(clSimulationKernel.advanceSimulationKernel, 1,
			                            sizeof(typeof(clSimulationKernel.particlesOutput)),
			                            &clSimulationKernel.particlesOutput);
			err |= clSetKernelArg(clSimulationKernel.advanceSimulationKernel, 2,
			                            sizeof(typeof(clSimulationKernel.collidedParticles)),
			                            &clSimulationKernel.collidedParticles);
			err |= clSetKernelArg(clSimulationKernel.advanceSimulationKernel, 3,
			                      sizeof(typeof(clSimulationKernel.minimumTime)),
			                      &clSimulationKernel.minimumTime);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set kernel arguments! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		size_t local;
		{ // Get the maximum work group size for executing the kernel on the device
			cl_int err = clGetKernelWorkGroupInfo(clSimulationKernel.advanceSimulationKernel, clState.device_id,
			                                      CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to retrieve kernel work group info! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Execute the kernel over the entire range of our 1d input data set using the maximum number of
			// work group items for this device
			size_t global = numberParticles;// TODO fix global group size
			local = 1;// TODO fix workgroup size
			cl_int err = clEnqueueNDRangeKernel(clState.commands, clSimulationKernel.advanceSimulationKernel, 1,
			                                    nullptr, &global, &local, 0, nullptr, nullptr);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				return EXIT_FAILURE;
			}
		}

		{ // Wait for the command commands to get serviced before reading back results
			clFinish(clState.commands); // TODO add to dependency list on the next read
		}
	}

	return EXIT_SUCCESS;
}

int main() {
	assert(numberParticles >= 2);

	srand(22);

	struct Particle particles[numberParticles];
	for (int i = 0; i < numberParticles; i++) {
		particles[i].position = generatePosition();
		particles[i].velocity = generateVelocity();
		printf("Create particle at (%f, %f)\n", particles[i].position.x, particles[i].position.y);
	}
	Time intersectionTimes[numberParticles * numberParticles]; // TODO make 2-dimensional array

	struct ClState clState = initClState(true);
	struct ClSimulationKernel clSimulationKernel = initSimulationKernel(clState, particles, particles, intersectionTimes);

	if(!clSimulationKernel.success) {
		return EXIT_FAILURE;
	}

	{ // Window initialization
		const int screenWidth = width;
		const int screenHeight = height;

		InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

		SetTargetFPS(60);
	}

	uint iteration = 0;
	bool paused = false;

	while (!WindowShouldClose()) {
		if(!paused) {
			for(int i = 0; i < numberParticles * numberParticles; i++) {
				intersectionTimes[i] = CL_INFINITY; // TODO check why this is necesary
			}

			{ // Write our data set into the input array in device memory
				cl_int err = clEnqueueWriteBuffer(clState.commands, clSimulationKernel.particlesInput, CL_TRUE, 0,
				                                  sizeof(typeof(*particles)) * numberParticles, particles, 0, nullptr,
				                                  &clSimulationKernel.writeParticlePositionsEvent);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to write to source array! %d\n", err);
					return EXIT_FAILURE;
				}
			}

			{ // Write our data set into the input array in device memory
				cl_int err = clEnqueueWriteBuffer(clState.commands, clSimulationKernel.intersectionTimes, CL_TRUE, 0,
				                                  sizeof(typeof(*intersectionTimes)) * numberParticles, intersectionTimes, 0, nullptr,
				                                  &clSimulationKernel.writeIntersectionTimesEvent);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to write to source array! %d\n", err);
					return EXIT_FAILURE;
				}
			}

			int err = callSimulation(clState, clSimulationKernel);

			if(err != EXIT_SUCCESS) {
				return EXIT_FAILURE;
			}

			{ // Read back the results from the device
				cl_int err = clEnqueueReadBuffer(clState.commands, clSimulationKernel.particlesOutput, CL_TRUE, 0,
				                                 sizeof(typeof(*particles)) * numberParticles, particles, 0,
				                                 nullptr, nullptr);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to read output array! %d\n", err);
					return EXIT_FAILURE;
				}
			}

			iteration++;
		}

		{
			BeginDrawing();

				ClearBackground(RAYWHITE);

				for (uint j = 0; j < numberParticles; j++) {
					DrawCircle((int) particles[j].position.x, (int) particles[j].position.y, 5, BLACK);
					DrawCircleLines((int) particles[j].position.x, (int) particles[j].position.y, radius, BLACK);
					DrawLine((int) particles[j].position.x, (int) particles[j].position.y,
							 (int) (particles[j].position.x + particles[j].velocity.x),
							 (int) (particles[j].position.y + particles[j].velocity.y), RED);

					char text[2048];
					snprintf(text, sizeof(text), "%u", j);
					DrawText(text, (int) particles[j].position.x + 5, (int) particles[j].position.y + 5, 11, BLACK);
				}

			EndDrawing();
		}

		if(IsKeyPressed(KEY_SPACE)) {
			paused = !paused;
		}
	}

	{ // Close window and OpenGL context
		CloseWindow();
	}

	{ // OpenCL shutdown and cleanup
		releaseClSimulationKernel(clSimulationKernel);
		releaseClState(clState);
	}

	return 0;
}