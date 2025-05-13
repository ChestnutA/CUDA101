//
// Created by Federico Vaccaro on 30/11/2018.
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <sstream>
#include <cmath>
#include <iomanip>
#include <stdlib.h>

#include "csvio.h"



int main() {
    const int nData = 500000;  // number of experiments

    std::default_random_engine generator;

    std::vector<std::normal_distribution<float>> centroids;

    std::normal_distribution<float> X0(3.0, 0.2);
    std::normal_distribution<float> Y0(3.0, 0.5);
    std::normal_distribution<float> Z0(3.0, 0.8);

    std::normal_distribution<float> X1(15.0, 1.0);
    std::normal_distribution<float> Y1(17.0, 0.2);
    std::normal_distribution<float> Z1(19.0, 0.6);

    std::normal_distribution<float> X2(10.5, 1.5);
    std::normal_distribution<float> Y2(0.0, 1.5);
    std::normal_distribution<float> Z2(5.0, 1.5);

    centroids.push_back(X0);
    centroids.push_back(Y0);
    centroids.push_back(Z0);
    centroids.push_back(X1);
    centroids.push_back(Y1);
    centroids.push_back(Z1);
    centroids.push_back(X2);
    centroids.push_back(Y2);
    centroids.push_back(Z2);

    std::vector<float> points;

    const int nCentroids = centroids.size()/3;

    points.resize(nData * 3);

    for (int i = 0; i < nData; ++i) {
        int centroid = i % nCentroids;
        std::normal_distribution<float> X_i = centroids[centroid*3];
        std::normal_distribution<float> Y_i = centroids[centroid*3 + 1];
        std::normal_distribution<float> Z_i = centroids[centroid*3 + 2];
        int idx = i * 3;
        points[idx] = X_i(generator);
        points[idx+1] = Y_i(generator);
        points[idx+2] = Z_i(generator);

    }
    std::string delimiter = ",";
    std::string filename = "dataset.csv";
    write2VecTo(filename.c_str(), delimiter, points);
    //read2VecFrom(filename.c_str(), delimiter, points);

    return 0;
}

