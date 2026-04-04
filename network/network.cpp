#include "network.h"

network::network() {}

std::vector<float> network::forward(const std::vector<float>& inputs) { return {}; }

void network::backward(const std::vector<float>& prediction, const std::vector<float>& actual) {}

void network::updateWeights(float learningRate) {}

float network::trainOneSample(const std::vector<float>& inputData, const std::vector<float>& actual) { return 0.0f; }

float network::trainBatch() { return 0.0f; }
