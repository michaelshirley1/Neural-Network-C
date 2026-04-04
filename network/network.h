#pragma once
#include <vector>
#include "layer.h"
#include "types.h"

struct layerConfig {
	int size;
	applyTypes activation;
	networkLayerType layerType;
};

class network {
public:
	network(const std::vector<layerConfig>& configs, float learningRate, lossType lossFn);

	std::vector<float> forward(const std::vector<float>& inputs);
	void backward(const std::vector<float>& prediction, const std::vector<float>& actual);
	void accumulateGradients();
	void updateWeights();

	float trainOneSample(const std::vector<float>& inputData, const std::vector<float>& actual);
	float trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& actuals);

	std::vector<layer> layers;
	float learningRate;
	lossType lossFn;
};
