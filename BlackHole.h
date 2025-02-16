#ifndef BLACKHOLE_H
#define BLACKHOLE_H

class BlackHole {
private:
    float mass;
    float3 position;  // Pozycja czarnej dziury
public:
    __host__ __device__ BlackHole(float m, float3 pos) : mass(m), position(pos) {}

    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ float3 getPosition() const { return position; }  // Funkcja zwracaj¹ca pozycjê czarnej dziury

    __host__ __device__ float getGravity(float r) const;  // Deklaracja funkcji
};

#endif // BLACKHOLE_H
