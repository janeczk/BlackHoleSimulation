#ifndef PHOTON_H
#define PHOTON_H

#include <cuda_runtime.h>

class Photon {
public:
    float3 position;
    float3 velocity;

    __host__ __device__ Photon() : position({ 0.0f, 0.0f, 0.0f }), velocity({ 0.0f, 0.0f, 0.0f }) {}

    __host__ __device__ Photon(float3 pos, float3 vel) : position(pos), velocity(vel) {}

    __host__ __device__ void update(float dt, float blackHoleMass, float3 blackHolePosition);  // Zmiana w funkcji update
};

#endif // PHOTON_H
