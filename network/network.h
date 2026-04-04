#pragma once
#include <vector>
#include "layer.h"

class network {
public:
	network();

	std::vector<float> forward(const std::vector<float>& inputs);
	void backward(const std::vector<float>& prediction, const std::vector<float>& actual);
	void updateWeights(float learningRate);

	float trainOneSample(const std::vector<float>& inputData, const std::vector<float>& actual);
	float trainBatch();

	std::vector<layer> layers;
};
