#include "BlackHole.h"
#include <cmath>

#define G 6.67430e-11f
#define EPSILON 1e-6f

__host__ __device__ void BlackHole::updatePosition(float dt, const BlackHole* blackHoles, int numHoles) {
    float3 totalAcceleration = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < numHoles; i++) {
        if (&blackHoles[i] == this) continue; // Pomijamy siebie

        float3 otherPos = blackHoles[i].getPosition();
        float dx = otherPos.x - position.x;
        float dy = otherPos.y - position.y;
        float r = sqrtf(dx * dx + dy * dy) + EPSILON;

        float force = G * blackHoles[i].getMass() / (r * r);
        totalAcceleration.x += force * (dx / r);
        totalAcceleration.y += force * (dy / r);
    }

    velocity.x += totalAcceleration.x * dt;
    velocity.y += totalAcceleration.y * dt;
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}
