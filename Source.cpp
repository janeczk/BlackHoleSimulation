#include <SFML/Graphics.hpp>
#include <chrono>
#include <vector>
#include <iostream>

#define SCALE 2
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define FIELD_WIDTH (WINDOW_WIDTH / SCALE)
#define FIELD_HEIGHT (WINDOW_HEIGHT / SCALE)

void computeField(uint8_t* result, float dt);
void cudaInit(size_t xSize, size_t ySize);
void cudaExit();

int main() {
    cudaInit(FIELD_WIDTH, FIELD_HEIGHT);
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Ray Tracing Black Hole");

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();

    sf::Texture texture;
    sf::Sprite sprite;
    std::vector<sf::Uint8> pixelBuffer(FIELD_WIDTH * FIELD_HEIGHT * 4);
    texture.create(FIELD_WIDTH, FIELD_HEIGHT);

    while (window.isOpen()) {
        end = std::chrono::system_clock::now();
        std::chrono::duration<float> diff = end - start;
        start = end;

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        computeField(pixelBuffer.data(), 0.005f);
        texture.update(pixelBuffer.data());
        sprite.setTexture(texture);
        sprite.setScale({ SCALE, SCALE });

        window.clear(sf::Color::Black);
        window.draw(sprite);
        window.display();
    }

    cudaExit();
    return 0;
}
