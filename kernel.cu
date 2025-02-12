#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define G 6.67430e-11
#define M 1.0
#define RS (2 * G * M)

__device__ void schwarzschild_geodesic(float3* pos, float3* vel, float dt) {
    float r = sqrt(pos->x * pos->x + pos->y * pos->y + pos->z * pos->z);
    if (r <= RS) return;

    float acc = -G * M / (r * r);
    vel->x += acc * pos->x / r * dt;
    vel->y += acc * pos->y / r * dt;
    vel->z += acc * pos->z / r * dt;

    pos->x += vel->x * dt;
    pos->y += vel->y * dt;
    pos->z += vel->z * dt;
}

__global__ void trace_rays(float3* positions, float3* velocities, int num_rays, float dt, int steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    for (int step = 0; step < steps; step++) {
        schwarzschild_geodesic(&positions[i], &velocities[i], dt);
    }
}

extern "C" __declspec(dllexport) void launch_kernel(float3* h_positions, float3* h_velocities, int num_rays, float dt, int steps) {
    float3* d_positions, * d_velocities;
    cudaMalloc(&d_positions, num_rays * sizeof(float3));
    cudaMalloc(&d_velocities, num_rays * sizeof(float3));

    cudaMemcpy(d_positions, h_positions, num_rays * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, h_velocities, num_rays * sizeof(float3), cudaMemcpyHostToDevice);

    trace_rays <<<(num_rays + 255) / 256, 256 >>> (d_positions, d_velocities, num_rays, dt, steps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_positions, d_positions, num_rays * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(d_positions);
    cudaFree(d_velocities);
}
