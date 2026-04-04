#include "helpers.h"
#include <ctime>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <cmath>

float helpers::randomUniform(float low, float high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(low, high);
    return distrib(gen);
}

float helpers::apply(applyTypes fn, float value) {
    switch (fn) {
        case applyTypes::RELU:
            return relu(value);
        case applyTypes::SIGMOID:
            return sigmoid(value);
        case applyTypes::TANH:
            return tanh(value);
        case applyTypes::LINEAR:
            return linear(value);
        default:
            return 0.0f;
    }
}

float helpers::relu(float value) {
    return std::max(0.0f, value);
}

float helpers::sigmoid(float value) {
    return 1 / (1 + std::exp(-value));
}

float helpers::tanh(float value) {
    return (std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value));
}

float helpers::linear(float value) {
    return value;
}

std::vector<float> helpers::softmax(std::vector<float> valueVector) {
    std::vector<float> result(valueVector.size());

    float maxVal = *std::max_element(valueVector.begin(), valueVector.end());

    float sum = 0.0f;
    for (float v : valueVector) {
        sum += std::exp(v - maxVal);
    }

    for (int i = 0; i < valueVector.size(); i++) {
        result[i] = std::exp(valueVector[i] - maxVal) / sum;
    }

    return result;
}

float helpers::activationDerivative(applyTypes activationFunction, float z) {
    switch (activationFunction) {
        case applyTypes::RELU:
            return reluDerivative(z);
        case applyTypes::SIGMOID:
            return sigmoidDerivative(z);
        case applyTypes::TANH:
            return tanhDerivative(z);
        case applyTypes::LINEAR:
            return linearDerivative(z);
        default:
            return 0.0f;
    }
}

float helpers::reluDerivative(float z) {
    return z > 0.0f ? 1.0f : 0.0f;
}

float helpers::sigmoidDerivative(float z) {
    float s = sigmoid(z);
    return s * (1.0f - s);
}

float helpers::tanhDerivative(float z) {
    float t = tanh(z);
    return 1.0f - t * t;
}

float helpers::linearDerivative(float z) {
    return 1.0f;
}
