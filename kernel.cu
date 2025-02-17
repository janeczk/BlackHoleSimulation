#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include <iostream>
#include <cstdlib>

#define SPAWN_INTERVAL 50  // Co ile klatek dodajemy nowe fotony
#define MAX_PHOTONS 50000   // Maksymalna liczba fotonów
#define PHOTONS_AT_ONCE 128  // Można zmieniać na dowolną liczbę, np. 8, 16, 32 itd.
using uint8_t = unsigned char;

struct Vec3 {
    float x, y, z;

    __device__ Vec3 operator+(Vec3 other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    __device__ Vec3 operator-(Vec3 other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    __device__ Vec3 operator*(float d) const {
        return { x * d, y * d, z * d };
    }
};

struct Photon {
    Vec3 position;
    Vec3 velocity;

    __device__ void update(float dt, Vec3 blackHolePos, float blackHoleMass) {
        Vec3 direction = blackHolePos - position;
        float distSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
        float force = blackHoleMass / (distSq + 0.001f); // Unikamy dzielenia przez 0
        velocity = velocity + direction * force * dt;
        position = position + velocity * dt;
    }
};

__constant__ Vec3 blackHolePos;
__constant__ float blackHoleMass;

Photon* d_photons;
uint8_t* colorField;
size_t numPhotons = 8;
size_t fieldWidth, fieldHeight;

#define CUDA_CALL(x) cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { std::cout << cudaGetErrorName(error) << std::endl; std::abort(); } x

void cudaInit(size_t width, size_t height) {
    fieldWidth = width;
    fieldHeight = height;
    cudaMalloc(&d_photons, MAX_PHOTONS * sizeof(Photon)); // Rezerwujemy miejsce na max. 5000 fotonów
    cudaMalloc(&colorField, fieldWidth * fieldHeight * 4 * sizeof(uint8_t));

    // Ustawienie stałej pozycji czarnej dziury
    Vec3 h_blackHolePos = { fieldWidth / 2.0f, fieldHeight / 2.0f, 0.0f };
    float h_blackHoleMass = 1.0e+4f;
    cudaMemcpyToSymbol(blackHolePos, &h_blackHolePos, sizeof(Vec3));
    cudaMemcpyToSymbol(blackHoleMass, &h_blackHoleMass, sizeof(float));
}



__global__ void updatePhotons(Photon* photons, size_t numPhotons, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPhotons) return;
    photons[i].update(dt, blackHolePos, blackHoleMass);
}

__global__ void renderPhotons(uint8_t* colorField, Photon* photons, size_t numPhotons, size_t width, size_t height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPhotons) return;

    int x = (int)photons[i].position.x;
    int y = (int)photons[i].position.y;
    if (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = (y * width + x) * 4;
        colorField[idx] = 255;     // Czerwony
        colorField[idx + 1] = 255;   // Zielony (zero)
        colorField[idx + 2] = 255;   // Niebieski (zero)
        colorField[idx + 3] = 255; // Alpha (pełna widoczność)
    }
}


void computeField(uint8_t* result, float dt, int frame) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((MAX_PHOTONS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    static int photonCount = 0;
    static int overwriteIndex = 0;

    // Co SPAWN_INTERVAL klatek dodajemy PHOTONS_AT_ONCE nowych fotonów
    if (frame % SPAWN_INTERVAL == 0) {
        Photon h_photons[PHOTONS_AT_ONCE];
        for (int i = 0; i < PHOTONS_AT_ONCE; i++) {
            float angle = (i * 2.0f * 3.14159265359f) / PHOTONS_AT_ONCE; // Równomierny rozkład wokół źródła

            h_photons[i].position = { fieldWidth / 2.0f, fieldHeight / 2.0f - 200.0f, 0.0f };
            h_photons[i].velocity = { cos(angle) * 300.0f, sin(angle) * 300.0f, 0.0f };
        }

        // Nadpisujemy stare fotony, jeśli osiągnęliśmy limit
        if (photonCount < MAX_PHOTONS) {
            cudaMemcpy(d_photons + photonCount, h_photons, PHOTONS_AT_ONCE * sizeof(Photon), cudaMemcpyHostToDevice);
            photonCount += PHOTONS_AT_ONCE;
        }
        else {
            cudaMemcpy(d_photons + overwriteIndex, h_photons, PHOTONS_AT_ONCE * sizeof(Photon), cudaMemcpyHostToDevice);
            overwriteIndex = (overwriteIndex + PHOTONS_AT_ONCE) % MAX_PHOTONS;
        }
    }

    // Aktualizacja wszystkich fotonów
    updatePhotons << <numBlocks, threadsPerBlock >> > (d_photons, photonCount, dt);
    cudaMemset(colorField, 50, fieldWidth * fieldHeight * 4 * sizeof(uint8_t)); // Ustawienie tła na szaro

    // Renderowanie wszystkich aktywnych fotonów
    renderPhotons << <numBlocks, threadsPerBlock >> > (colorField, d_photons, photonCount, fieldWidth, fieldHeight);
    cudaMemcpy(result, colorField, fieldWidth * fieldHeight * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}






void cudaExit() {
    cudaFree(d_photons);
    cudaFree(colorField);
}