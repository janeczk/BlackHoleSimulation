#include "BlackHole.h"
#include "device_launch_parameters.h"

#define G 6.67430e-11f
#define EPSILON 1e-6f  // Aby unikn¹æ dzielenia przez 0

__host__ __device__ float BlackHole::getGravity(float r) const {
    return G * mass / ((r + EPSILON) * (r + EPSILON));
}
