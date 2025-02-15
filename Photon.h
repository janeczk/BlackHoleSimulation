#ifndef PHOTON_H
#define PHOTON_H

#include <cuda_runtime.h>

class Photon {
public:
    float3 position;
    float3 velocity;

    // Domyœlny konstruktor (potrzebny do alokacji tablicy)
    __host__ __device__ Photon() : position({ 0.0f, 0.0f, 0.0f }), velocity({ 0.0f, 0.0f, 0.0f }) {}

    // Konstruktor inicjalizuj¹cy foton na danej pozycji z dan¹ prêdkoœci¹
    __host__ __device__ Photon(float3 pos, float3 vel) : position(pos), velocity(vel) {}

    // Aktualizacja pozycji i prêdkoœci pod wp³ywem grawitacji czarnej dziury
    __host__ __device__ void update(float dt, float blackHoleMass);
};

#endif // PHOTON_H
