﻿#include "SFML/Graphics.hpp"
#include "Photon.h"
#include "BlackHole.h"
#include <iostream>
#include <random>

#define WIN_SIZE 1200
#define NUM_RAYS 2000
#define DT 0.002f
#define RADIUS 150.0f
#define PI 3.14159265359f
#define NOISE 20.0f
#define VELOCITY_NOISE 30.0f
#define VELOCITY 500.0f
#define NUM_BLACK_HOLES 2  // Liczba czarnych dziur

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-NOISE, NOISE);
    std::uniform_real_distribution<float> vel_noise_dist(-VELOCITY_NOISE, VELOCITY_NOISE);

    Photon* photons = new Photon[NUM_RAYS];

    // Pozycje czarnych dziur
    BlackHole blackHoles[NUM_BLACK_HOLES] = {
        BlackHole(5.23123e+17f, { -200.0f, 0.0f, 0.0f }), // Pierwsza czarna dziura
        BlackHole(5.23123e+17f, { 200.0f, 0.0f, 0.0f })   // Druga czarna dziura
    };

    for (int i = 0; i < NUM_RAYS; i++) {
        float angle = (2 * PI * i) / NUM_RAYS;
        float3 pos = { RADIUS * cos(angle) + noise_dist(gen), RADIUS * sin(angle) + noise_dist(gen), 0.0f };
        float3 vel = { -sin(angle) * VELOCITY + vel_noise_dist(gen), cos(angle) * VELOCITY + vel_noise_dist(gen), 0.0f };
        photons[i] = Photon(pos, vel);
    }

    sf::RenderWindow window(sf::VideoMode(WIN_SIZE, WIN_SIZE), "Schwarzschild Ray Tracing");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color(10, 10, 30));

        // Rysowanie czarnych dziur
        for (int i = 0; i < NUM_BLACK_HOLES; i++) {
            blackHoles[i].updatePosition(DT, blackHoles, NUM_BLACK_HOLES);

            sf::CircleShape blackHoleShape(10);
            blackHoleShape.setOrigin(10, 10);
            blackHoleShape.setPosition(WIN_SIZE / 2 + blackHoles[i].getPosition().x, WIN_SIZE / 2 + blackHoles[i].getPosition().y);
            blackHoleShape.setFillColor(sf::Color::Yellow);
            window.draw(blackHoleShape);
        }

        // Aktualizacja fotonów
        for (int i = 0; i < NUM_RAYS; i++) {
            photons[i].update(DT, blackHoles, NUM_BLACK_HOLES);
            sf::CircleShape pixel(1);
            pixel.setPosition(WIN_SIZE / 2 + photons[i].position.x, WIN_SIZE / 2 + photons[i].position.y);
            pixel.setFillColor(sf::Color::White);
            window.draw(pixel);
        }

        window.display();
    }

    delete[] photons;
    return 0;
}

