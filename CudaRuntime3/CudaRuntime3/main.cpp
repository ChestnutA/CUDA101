/*
 ============================================================================
 Name        : MeanShiftClustering.cu
 Author      : Lorenzo Agnolucci
 Version     :
 Copyright   :
 Description : CUDA implementation of Mean Shift clustering algorithm
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <chrono>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define NUM_ITERATIONS 10
#define NUM_TESTS 1
#define BANDWIDTH 1.5f

std::vector<float> readPointsFromCSV(const std::string& fileName) {
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::ifstream data(fileName);
	std::cout << "test: " << data.fail() << std::endl;
	std::string line;
	while (std::getline(data, line)) {
		std::vector<float> point;
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			point.push_back(stod(cell));
		}
		x.push_back(point[0]);
		x.push_back(point[1]);
		x.push_back(point[2]);
	}
	return x;
	std::vector<float> points = x;
	points.insert(points.end(), y.begin(), y.end());
	points.insert(points.end(), z.begin(), z.end());
	return points;
}

struct float3 {
    float x, y, z;
    float3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
    float3 operator+(const float3& other) const {
        return float3(x + other.x, y + other.y, z + other.z);
    }
    float3 operator-(const float3& other) const {
        return float3(x - other.x, y - other.y, z - other.z);
    }
    float3& operator+=(const float3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    float3 operator*(float scalar) const {
        return float3(x * scalar, y * scalar, z * scalar);
    }
    float3 operator/(float scalar) const {
        return float3(x / scalar, y / scalar, z / scalar);
    }
};

float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void naiveMeanShift(float* shiftedPoints, const float* originalPoints, unsigned numPoints) {
    #pragma omp parallel for
    for (unsigned idx = 0; idx < numPoints; ++idx) {
        float x = shiftedPoints[idx];
        float y = shiftedPoints[idx + numPoints];
        float z = shiftedPoints[idx + 2 * numPoints];
        float3 currentPoint(x, y, z);

        float3 newPosition(0.0f, 0.0f, 0.0f);
        float totalWeight = 0.0f;

        for (unsigned i = 0; i < numPoints; ++i) {
            float ox = originalPoints[i];
            float oy = originalPoints[i + numPoints];
            float oz = originalPoints[i + 2 * numPoints];
            float3 point(ox, oy, oz);

            float3 diff = currentPoint - point;
            float squaredDist = dot(diff, diff);
            float weight = std::exp(-squaredDist / (2.0f * BANDWIDTH * BANDWIDTH));

            newPosition += point * weight;
            totalWeight += weight;
        }

        if (totalWeight > 1e-7f) {
            newPosition = newPosition / totalWeight;
        }

        shiftedPoints[idx] = newPosition.x;
        shiftedPoints[idx + numPoints] = newPosition.y;
        shiftedPoints[idx + 2 * numPoints] = newPosition.z;
    }
}


int main(void)
{

	std::string fileName = "datasets/different_size/3D_data_250000.csv";
	std::vector<float> inputPoints = readPointsFromCSV(fileName);

	int numPoints = inputPoints.size() / 3;
	std::cout << "Num points: " << numPoints << std::endl;

	float totalElapsedTime = 0.0;

	std::vector<float> originalPoints = inputPoints;
	std::vector<float> shiftedPoints = inputPoints;

	for (int j = 0; j < NUM_TESTS; j++) {
		originalPoints = inputPoints;
		shiftedPoints = inputPoints;

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            naiveMeanShift(shiftedPoints.data(), originalPoints.data(), numPoints);
        }

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		float elapsedTime = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
		totalElapsedTime += elapsedTime;
	}
	totalElapsedTime /= NUM_TESTS;
	std::cout << "\nTiling Mean Shift elapsed time: " << totalElapsedTime<< std::endl;
	for (int i = numPoints; i < numPoints + 1; i++) {
		std::cout << shiftedPoints[i] << ",";
		std::cout << shiftedPoints[i + 1] << ",";
		std::cout << shiftedPoints[i + 2] << "\n";
	}
	return 0;
}
