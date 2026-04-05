#include "neuron.h"
#include <vector>
#include <math-helper/helpers.h>

neuron::neuron(int numInputs, applyTypes initActivationFunction) {
	activationFunction = initActivationFunction;

	weights = std::vector<weight>();

	for (int i = 0; i < numInputs; i++) {
		float scale = 1.0f / sqrt((float)numInputs);
		float value = helpers::randomUniform(-scale, scale);

		weights.push_back(
			weight(value)
		);
	}

	bias = weight(0.0);

	rawOutput = 0.0;
	activatedOutput = 0.0;
	delta = 0.0;
}

float neuron::forward(const std::vector<float>& inputs) {
	rawOutput = bias.value;

	for (int i = 0; i < inputs.size(); i++) {
		rawOutput += weights[i].value * inputs[i];
	}

	activatedOutput = helpers::apply(activationFunction, rawOutput);

	return activatedOutput;
}

void neuron::computeDelta(float downstreamError) {
	delta = downstreamError * helpers::activationDerivative(activationFunction, rawOutput);
}

void neuron::accumulateGradients(const std::vector<float>& inputs) {
	for (int i = 0; i < inputs.size(); i++) {
		weights[i].gradient += delta * inputs[i];
	}
	bias.gradient += delta;
}

void neuron::updateWeights(float learningRate) {
	for (int i = 0; i < weights.size(); i++) {
		weight& currentWeight = weights[i];
		currentWeight -= learningRate * currentWeight.gradient;
		currentWeight.resetGradient();
	}
	bias -= learningRate * bias.gradient;
	bias.resetGradient();
}
