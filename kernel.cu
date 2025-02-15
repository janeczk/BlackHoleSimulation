#include <cuda_runtime.h>
#include <cmath>
#include "device_launch_parameters.h"

#define G 6.67430e-11  // Sztuczna "stała grawitacyjna" (większa wartość = mocniejsze przyciąganie)
#define BLACK_HOLE_MASS 5.23123e+17  
#define EPSILON 1e-6f  // Minimalna odległość od środka, żeby uniknąć dzielenia przez 0

__global__ void updateKernel(float3* positions, float3* velocities, int num_rays, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    float3 pos = positions[i];
    float3 vel = velocities[i];

    // Obliczanie odległości od czarnej dziury (zakładamy, że jest w (0,0))
    float r = sqrtf(pos.x * pos.x + pos.y * pos.y) + EPSILON; 

    // Przyciąganie grawitacyjne: siła działa w kierunku środka (0,0)
    float force = G * BLACK_HOLE_MASS / (r * r);
    float3 acceleration = { -force * (pos.x / r), -force * (pos.y / r), 0.0f };

    // Aktualizacja prędkości i pozycji
    vel.x += acceleration.x * dt;
    vel.y += acceleration.y * dt;
    
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;

    // Zapis wyników
    positions[i] = pos;
    velocities[i] = vel;
}

extern "C" void update_positions(float3* d_positions, float3* d_velocities, float3* h_positions, int num_rays, float dt) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;

    updateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities, num_rays, dt);

    // Pobranie nowych pozycji z GPU do CPU
    cudaMemcpy(h_positions, d_positions, num_rays * sizeof(float3), cudaMemcpyDeviceToHost);
}
