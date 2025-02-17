#include "LightSource.h"
#include <math.h>

__host__ __device__ LightSource::LightSource() {
    position = make_float3(0.0f, 0.0f, 500.0f);
    numDirections = 8;
}

__host__ __device__ LightSource::LightSource(float3 pos, int directions) {
    position = pos;
    numDirections = directions;
}

__host__ __device__ float3 LightSource::getPosition() const {
    return position;
}

__global__ void generatePhotons(Photon* photons, float3 lightPos, int numDirections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numDirections) return;

    float angle = (2.0f * 3.14159265359f * i) / numDirections;
    float3 pos = lightPos;
    float3 vel = make_float3(cos(angle) * 500.0f, sin(angle) * 500.0f, 0.0f);

    photons[i] = Photon(pos, vel);
}

void createLightSource(Photon* d_photons, int numDirections) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((numDirections + threadsPerBlock.x - 1) / threadsPerBlock.x);

    float3 lightPos = make_float3(0.0f, -300.0f, 0.0f);
    generatePhotons << <numBlocks, threadsPerBlock >> > (d_photons, lightPos, numDirections);
    cudaDeviceSynchronize();
}
