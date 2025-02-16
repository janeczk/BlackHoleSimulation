#include "Photon.h"
#include <cmath>

#define G 6.67430e-11f
#define EPSILON 1e-6f

__host__ __device__ void Photon::update(float dt, float blackHoleMass, float3 blackHolePosition) {
    // Obliczanie wektora odleg³oœci miêdzy fotonem a czarn¹ dziur¹
    float dx = position.x - blackHolePosition.x;
    float dy = position.y - blackHolePosition.y;

    // Obliczanie odleg³oœci r miêdzy fotonem a czarn¹ dziur¹
    float r = sqrtf(dx * dx + dy * dy) + EPSILON;

    // Przyci¹ganie grawitacyjne - obliczanie si³y przyci¹gania
    float force = G * blackHoleMass / (r * r);

    // Obliczanie przyspieszenia fotonu na podstawie si³y grawitacyjnej
    float3 acceleration = { -force * (dx / r), -force * (dy / r), 0.0f };

    // Aktualizacja prêdkoœci na podstawie przyspieszenia
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;

    // Aktualizacja pozycji fotonu na podstawie prêdkoœci
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}
