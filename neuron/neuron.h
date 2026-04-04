#pragma once
#include <vector>
#include "weight.h"
#include "types.h"

class neuron {
public:
	neuron(int numInputs, applyTypes initActivationFunction);

	virtual float forward(const std::vector<float>& inputs);
	void computeDelta(float downstreamError);
	void accumulateGradients(const std::vector<float>& inputs);
	void updateWeights(float learningRate);

	std::vector<weight> weights;
	weight bias = 0.0f;
	float rawOutput = 0.0f;
	float activatedOutput = 0.0f;
	float delta = 0.0f;
	applyTypes activationFunction;
};

class inputNeuron : public neuron {
public:
	inputNeuron() : neuron(0, applyTypes::LINEAR) {
		rawOutput = 0.0f;
		activatedOutput = 0.0f;
		delta = 0.0f;
	}

	float forward(float value) {
		activatedOutput = value;
		return activatedOutput;
	}

	float forward(const std::vector<float>& inputs) override {
		return forward(inputs[0]);
	}
};
