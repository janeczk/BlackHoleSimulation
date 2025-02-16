#include "Camera.h"
#include <cmath>

#define PI 3.14159265359f

Camera::Camera(sf::Vector3f startPos) {
    position = { startPos.x, startPos.y, startPos.z };
    front = { 0.0f, 0.0f, -1.0f };  // Domyœlnie skierowana wzd³u¿ osi -Z
    up = { 0.0f, 1.0f, 0.0f };      // Domyœlnie skierowana wzd³u¿ osi +Y

    yaw = -90.0f;  // Pocz¹tkowa wartoœæ, aby kamera patrzy³a na -Z
    pitch = 0.0f;
}

void Camera::processMouseMovement(float offsetX, float offsetY) {
    float sensitivity = 0.1f; // Mo¿esz dostosowaæ czu³oœæ
    offsetX *= sensitivity;
    offsetY *= sensitivity;

    yaw += offsetX;
    pitch += offsetY;

    // Ograniczenie pitch do [-89, 89], aby unikn¹æ odwrócenia kamery
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    // Przeliczenie wektora kierunku
    float3 direction;
    direction.x = cos(yaw * (PI / 180.0f)) * cos(pitch * (PI / 180.0f));
    direction.y = sin(pitch * (PI / 180.0f));
    direction.z = sin(yaw * (PI / 180.0f)) * cos(pitch * (PI / 180.0f));

    // Normalizacja wektora front
    float length = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    front = { direction.x / length, direction.y / length, direction.z / length };
}

float2 Camera::project3DTo2D(float3 worldPos) {
    float fov = 90.0f; // Pole widzenia
    float aspectRatio = 1.0f;
    float nearPlane = 0.1f;

    float scale = tan((fov * 0.5f) * (PI / 180.0f));

    // Przesuniêcie pozycji œwiata wzglêdem kamery
    float3 relativePos = {
        worldPos.x - position.x,
        worldPos.y - position.y,
        worldPos.z - position.z
    };

    float z = relativePos.x * front.x + relativePos.y * front.y + relativePos.z * front.z;

    if (z <= nearPlane) z = nearPlane; // Unikniêcie dzielenia przez 0

    float x2D = (relativePos.x / (z * scale)) * 600.0f + 600.0f;
    float y2D = (relativePos.y / (z * scale)) * 600.0f + 600.0f;

    return { x2D, y2D };
}
