#ifndef PHOTON_H
#define PHOTON_H

#include <cuda_runtime.h>

class Photon {
public:
    float3 position;
    float3 velocity;

    // Domy�lny konstruktor (potrzebny do alokacji tablicy)
    __host__ __device__ Photon() : position({ 0.0f, 0.0f, 0.0f }), velocity({ 0.0f, 0.0f, 0.0f }) {}

    // Konstruktor inicjalizuj�cy foton na danej pozycji z dan� pr�dko�ci�
    __host__ __device__ Photon(float3 pos, float3 vel) : position(pos), velocity(vel) {}

    // Aktualizacja pozycji i pr�dko�ci pod wp�ywem grawitacji czarnej dziury
    __host__ __device__ void update(float dt, float blackHoleMass);
};

#endif // PHOTON_H
