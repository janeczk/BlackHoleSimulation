#include "Photon.h"
#include <cmath>

#define G 6.67430e-11f
#define EPSILON 1e-6f

__host__ __device__ void Photon::update(float dt, BlackHole* blackHoles, int numBlackHoles) {
    float3 totalAcceleration = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < numBlackHoles; i++) {
        float3 blackHolePos = blackHoles[i].getPosition();
        float blackHoleMass = blackHoles[i].getMass();

        float dx = position.x - blackHolePos.x;
        float dy = position.y - blackHolePos.y;
        float r = sqrtf(dx * dx + dy * dy) + EPSILON;

        float force = G * blackHoleMass / (r * r);
        totalAcceleration.x += -force * (dx / r);
        totalAcceleration.y += -force * (dy / r);
    }

    velocity.x += totalAcceleration.x * dt;
    velocity.y += totalAcceleration.y * dt;
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}
