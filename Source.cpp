#include "SFML/Graphics.hpp"
#include "SFML/Window.hpp"
#include "SFML/System.hpp"
#include "Photon.h"
#include "BlackHole.h"
#include "Camera.h"
#include <iostream>
#include <random>

#define WIN_SIZE 1200
#define NUM_RAYS 2000
#define DT 0.0006f
#define RADIUS 400.0f
#define PI 3.14159265359f
#define NOISE 20.0f
#define VELOCITY_NOISE 200.0f
#define VELOCITY 500.0f
#define NUM_BLACK_HOLES 3// Liczba czarnych dziur
#define SCALE 0.5

sf::Vector2i lastMousePos;
bool mouseHeld = false;

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-NOISE, NOISE);
    std::uniform_real_distribution<float> vel_noise_dist(-VELOCITY_NOISE, VELOCITY_NOISE);

    Photon* photons = new Photon[NUM_RAYS];
    
    Camera camera({ 0.0f, 0.0f, 1000.0f });

    // Pozycje czarnych dziur
    BlackHole blackHoles[NUM_BLACK_HOLES] = {
        BlackHole(5.23123e+17f, { -200.0f, 0.0f, 0.0f }),
        BlackHole(5.23123e+17f, { 200.0f, 0.0f, 0.0f }),
        BlackHole(5.23123e+19f, { 100.0f, 100.0f, 0.0f })   // Druga czarna dziura
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

            // Aktywacja myszy po kliknięciu
            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                mouseHeld = true;
                lastMousePos = sf::Mouse::getPosition(window);
            }

            if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
                mouseHeld = false;
            }

            // Obsługa ruchu myszy
            if (event.type == sf::Event::MouseMoved && mouseHeld) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                float offsetX = mousePos.x - lastMousePos.x;
                float offsetY = mousePos.y - lastMousePos.y;
                lastMousePos = mousePos;

                camera.processMouseMovement(offsetX, offsetY);

                std::cout << "Yaw: " << camera.yaw << " Pitch: " << camera.pitch << std::endl;
            }
        }
        window.clear(sf::Color(10, 10, 30));

        // Rysowanie czarnych dziur
        for (int i = 0; i < NUM_BLACK_HOLES; i++) {
            blackHoles[i].updatePosition(DT, blackHoles, NUM_BLACK_HOLES);

            float2 screenPos = camera.project3DTo2D(blackHoles[i].getPosition());

            sf::CircleShape blackHoleShape(10);
            blackHoleShape.setOrigin(10, 10);
            blackHoleShape.setPosition(screenPos.x,screenPos.y);
            blackHoleShape.setFillColor(sf::Color::Yellow);
            window.draw(blackHoleShape);
        }

        // Aktualizacja i rysowanie fotonów
        for (int i = 0; i < NUM_RAYS; i++) {
            photons[i].update(DT, blackHoles, NUM_BLACK_HOLES);

            float2 screenPos = camera.project3DTo2D(photons[i].position);

            sf::CircleShape pixel(1);
            pixel.setPosition(screenPos.x,screenPos.y);
            pixel.setFillColor(sf::Color::White);
            window.draw(pixel);
        }


        window.display();
    }

    delete[] photons;
    return 0;
}

