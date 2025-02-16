#ifndef BLACKHOLE_H
#define BLACKHOLE_H

#include <cuda_runtime.h>

class BlackHole {
private:
    float mass;
    float3 position;
    float3 velocity; // Dodaj pr�dko��

public:
    __host__ __device__ BlackHole(float m, float3 pos) : mass(m), position(pos), velocity({ 0.0f, 0.0f, 0.0f }) {}

    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ float3 getPosition() const { return position; }
    __host__ __device__ void setVelocity(float3 vel) { velocity = vel; }
    __host__ __device__ float3 getVelocity() const { return velocity; }

    // Nowa metoda aktualizacji pozycji na podstawie si�y grawitacyjnej
    __host__ __device__ void updatePosition(float dt, const BlackHole* blackHoles, int numHoles);
};

#endif // BLACKHOLE_H
