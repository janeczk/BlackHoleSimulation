#ifndef CAMERA_H
#define CAMERA_H

#include <SFML/System/Vector3.hpp>
#include "SFML/System.hpp"
#include <cuda_runtime.h>

class Camera {
public:
    float3 position;
    float3 front;
    float3 up;

    float yaw;
    float pitch;

    Camera(sf::Vector3f startPos);

    void processMouseMovement(float offsetX, float offsetY);
    float2 project3DTo2D(float3 worldPos);
};

#endif // CAMERA_H
