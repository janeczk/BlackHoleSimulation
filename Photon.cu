#include "Photon.h"
#include <cmath>

#define G 6.67430e-11f  // Sztuczna "sta³a grawitacyjna"
#define EPSILON 1e-6f   // Minimalna odleg³oœæ od œrodka (unikanie dzielenia przez zero)

__host__ __device__ void Photon::update(float dt, float blackHoleMass) {
    float r = sqrtf(position.x * position.x + position.y * position.y) + EPSILON;

    // Przyci¹ganie grawitacyjne
    float force = G * blackHoleMass / (r * r);
    float3 acceleration = { -force * (position.x / r), -force * (position.y / r), 0.0f };

    // Aktualizacja prêdkoœci i pozycji
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}
