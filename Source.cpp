#include "SFML/Graphics.hpp"
#include <iostream>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

// Deklaracje funkcji CUDA
extern "C" __declspec(dllimport) void launch_kernel(float3* h_positions, float3* h_velocities, int num_rays, float dt, int steps);


const int num_rays = 1000;
const float dt = 0.01f;
const int steps = 1000;

int main() {
    float3* h_positions = new float3[num_rays];
    float3* h_velocities = new float3[num_rays];

    for (int i = 0; i < num_rays; i++) {
        h_positions[i] = { 1.5f, 0.0f, 0.0f };
        h_velocities[i] = { 0.0f, 1.0f, 0.0f };
    }

    // Uruchomienie kernela CUDA
    launch_kernel(h_positions, h_velocities, num_rays, dt, steps);

    // Tworzenie okna SFML
    sf::RenderWindow window(sf::VideoMode(800, 800), "Schwarzschild Ray Tracing");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        for (int i = 0; i < num_rays; i++) {
            sf::CircleShape pixel(1);
            std::cout << "Photon " << i << " Position: " << h_positions[i].x << ", " << h_positions[i].y << std::endl;
            const float scale = 200.0f;  // Skalowanie pozycji do ekranu
            pixel.setPosition(400 + h_positions[i].x * scale, 400 + h_positions[i].y * scale);
            pixel.setFillColor(sf::Color::White);
            window.draw(pixel);
        }
        window.display();
    }

    delete[] h_positions;
    delete[] h_velocities;
    return 0;
}
