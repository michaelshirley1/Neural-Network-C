#include "layer.h"
#include <math-helper/helpers.h>

layer::layer(int numNeurons, int numInputsPerNeuron, applyTypes initActivationFunction, networkLayerType initLayerType) {
	layerType = initLayerType;
	activationFunction = initActivationFunction;

	if (layerType == networkLayerType::INPUT) {
		for (int i = 0; i < numNeurons; i++) {
			neurons.push_back(inputNeuron());
		}
	}
	else {
		for (int i = 0; i < numNeurons; i++) {
			neurons.push_back(neuron(numInputsPerNeuron, initActivationFunction));
		}
	}
}

std::vector<float> layer::getOutputs() {
	std::vector<float> outputs;
	for (auto& n : neurons) {
		outputs.push_back(n.activatedOutput);
	}
	return outputs;
}

void layer::updateWeights(float learningRate) {
	for (auto& n : neurons) {
		n.updateWeights(learningRate);
	}
}

std::vector<float> layer::forward(const std::vector<float>& inputs) {
	if (layerType == networkLayerType::INPUT) {
		for (int i = 0; i < neurons.size(); i++) {
			neurons[i].activatedOutput = inputs[i];
		}
		return inputs;
	}

	if (activationFunction == applyTypes::SOFTMAX) {
		for (auto& n : neurons) {
			n.rawOutput = n.bias.value;
			for (int j = 0; j < inputs.size(); j++) {
				n.rawOutput += n.weights[j].value * inputs[j];
			}
		}

		std::vector<float> rawOutputs;
		for (auto& n : neurons) {
			rawOutputs.push_back(n.rawOutput);
		}

		std::vector<float> softmaxVals = helpers::softmax(rawOutputs);

		for (int i = 0; i < neurons.size(); i++) {
			neurons[i].activatedOutput = softmaxVals[i];
		}
	}
	else {
		for (auto& n : neurons) {
			n.forward(inputs);
		}
	}

	return getOutputs();
}

std::vector<float> layer::backward(const std::vector<float>& downstreamErrors) {
	for (int i = 0; i < neurons.size(); i++) {
		neurons[i].computeDelta(downstreamErrors[i]);
	}

	std::vector<float> prevErrors(neurons[0].weights.size(), 0.0f);
	for (int i = 0; i < neurons.size(); i++) {
		for (int j = 0; j < neurons[i].weights.size(); j++) {
			prevErrors[j] += neurons[i].weights[j].value * neurons[i].delta;
		}
	}

	return prevErrors;
}

void layer::accumulateGradients(const std::vector<float>& inputs) {
	for (auto& n : neurons) {
		n.accumulateGradients(inputs);
	}
}
