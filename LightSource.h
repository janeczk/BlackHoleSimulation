#ifndef LIGHTSOURCE_H
#define LIGHTSOURCE_H

#include <cuda_runtime.h>
#include "Photon.h"
#include "BlackHole.h"
#include <cmath>

class LightSource {
public:
    float3 position;
    int numDirections;

    __host__ __device__ LightSource();
    __host__ __device__ LightSource(float3 pos, int directions);

    __host__ __device__ float3 getPosition() const;
};

// Funkcje do CUDA
void createLightSource(Photon* d_photons, int numDirections);
//void updatePhotons(Photon* d_photons, BlackHole* d_blackHoles, int numBlackHoles, float dt);

#endif // LIGHTSOURCE_H
