#pragma once
#include <vector>
#include "types.h"
#include "neuron.h"

class layer {
public:
	layer(int numNeurons, int numInputsPerNeuron, applyTypes activationFunction, networkLayerType layerType);

	std::vector<float> getOutputs();
	void updateWeights(float learningRate);

	std::vector<float> forward(const std::vector<float>& inputs);
	std::vector<float> backward(const std::vector<float>& downstreamErrors);

	void accumulateGradients(const std::vector<float>& inputs);

	networkLayerType layerType;
	applyTypes activationFunction;
	std::vector<neuron> neurons;
};
