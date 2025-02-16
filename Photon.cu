#include "Photon.h"
#include <cmath>

#define G 6.67430e-11f
#define EPSILON 1e-6f

__host__ __device__ void Photon::update(float dt, float blackHoleMass, float3 blackHolePosition) {
    // Obliczanie wektora odleg�o�ci mi�dzy fotonem a czarn� dziur�
    float dx = position.x - blackHolePosition.x;
    float dy = position.y - blackHolePosition.y;

    // Obliczanie odleg�o�ci r mi�dzy fotonem a czarn� dziur�
    float r = sqrtf(dx * dx + dy * dy) + EPSILON;

    // Przyci�ganie grawitacyjne - obliczanie si�y przyci�gania
    float force = G * blackHoleMass / (r * r);

    // Obliczanie przyspieszenia fotonu na podstawie si�y grawitacyjnej
    float3 acceleration = { -force * (dx / r), -force * (dy / r), 0.0f };

    // Aktualizacja pr�dko�ci na podstawie przyspieszenia
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;

    // Aktualizacja pozycji fotonu na podstawie pr�dko�ci
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}
