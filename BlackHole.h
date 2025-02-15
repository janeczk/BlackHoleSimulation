#ifndef BLACKHOLE_H
#define BLACKHOLE_H

class BlackHole {
private:
    float mass;
public:
    __host__ __device__ BlackHole(float m) : mass(m) {}

    __host__ __device__ float getMass() const { return mass; }

    __host__ __device__ float getGravity(float r) const;  // Deklaracja funkcji
};

#endif // BLACKHOLE_H
