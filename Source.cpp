#include "SFML/Graphics.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

#define WIN_SIZE 1200
#define NUM_RAYS 2000  
#define DT 0.005f       //  Spowolniona symulacja
#define RADIUS 150.0f  // Promień startowy fotonów wokół czarnej dziury
#define PI 3.14159265359f
#define NOISE 30.0f  //  Intensywność szumu pozycji startowej
#define VELOCITY_NOISE 30.0f  //  Szum do prędkości startowej
#define VELOCITY 500.0f

// Deklaracja funkcji CUDA
extern "C" void update_positions(float3* d_positions, float3* d_velocities, float3* h_positions, int num_rays, float dt);

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-NOISE, NOISE);
    std::uniform_real_distribution<float> vel_noise_dist(-VELOCITY_NOISE, VELOCITY_NOISE);

    float3* h_positions = new float3[NUM_RAYS];
    float3* h_velocities = new float3[NUM_RAYS];

    // Umieszczamy fotony na okręgu wokół czarnej dziury z losowym szumem
    for (int i = 0; i < NUM_RAYS; i++) {
        float angle = (2 * PI * i) / NUM_RAYS;
        float noise_x = noise_dist(gen);
        float noise_y = noise_dist(gen);

        h_positions[i] = { RADIUS * cos(angle) + noise_x, RADIUS * sin(angle) + noise_y, 0.0f };

        // Prędkość początkowa: ruch okrężny + losowy szum
        float vx = -sin(angle) * VELOCITY + vel_noise_dist(gen);
        float vy = cos(angle) * VELOCITY + vel_noise_dist(gen);
        h_velocities[i] = { vx, vy, 0.0f };
    }

    // Alokacja pamięci GPU
    float3* d_positions, * d_velocities;
    cudaMalloc(&d_positions, NUM_RAYS * sizeof(float3));
    cudaMalloc(&d_velocities, NUM_RAYS * sizeof(float3));

    cudaMemcpy(d_positions, h_positions, NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, h_velocities, NUM_RAYS * sizeof(float3), cudaMemcpyHostToDevice);

    sf::RenderWindow window(sf::VideoMode(WIN_SIZE, WIN_SIZE), "Schwarzschild Ray Tracing");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        //AKTUALIZACJA CUDA
        update_positions(d_positions, d_velocities, h_positions, NUM_RAYS, DT);

        // Rysowanie
        window.clear(sf::Color(10, 10, 30)); // Ciemnogranatowe tło

        // Czarna dziura (duży czarny okrąg w środku)
        sf::CircleShape blackHole(10);
        blackHole.setOrigin(10, 10);
        blackHole.setPosition(WIN_SIZE/2, WIN_SIZE/2);
        blackHole.setFillColor(sf::Color::Black);
        window.draw(blackHole);

        // Fotony
        for (int i = 0; i < NUM_RAYS; i++) {
            sf::CircleShape pixel(1);
            pixel.setPosition(WIN_SIZE/2 + h_positions[i].x, WIN_SIZE/2 + h_positions[i].y);
            pixel.setFillColor(sf::Color::White);
            window.draw(pixel);
        }

        window.display();
    }

    // Zwolnienie pamięci
    cudaFree(d_positions);
    cudaFree(d_velocities);
    delete[] h_positions;
    delete[] h_velocities;

    return 0;
}
