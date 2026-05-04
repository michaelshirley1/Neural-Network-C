#include "layer.h"
#include <math-helper/helpers.h>
#include <algorithm>
#include <execution>

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
	std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
		outputs.push_back(n.activatedOutput);
	});
	return outputs;
}

void layer::updateWeights(float learningRate) {
	std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
		n.updateWeights(learningRate);
	});
}

std::vector<float> layer::forward(const std::vector<float>& inputs) {
	if (layerType == networkLayerType::INPUT) {
		for (int i = 0; i < neurons.size(); i++) {
			neurons[i].activatedOutput = inputs[i];
		}
		return inputs;
	}

	if (activationFunction == applyTypes::SOFTMAX) {
		std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
			n.rawOutput = n.bias.value;
			for (int j = 0; j < inputs.size(); j++) {
				n.rawOutput += n.weights[j].value * inputs[j];
			}
		});

		std::vector<float> rawOutputs;
		std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
			rawOutputs.push_back(n.rawOutput);
		});

		std::vector<float> softmaxVals = helpers::softmax(rawOutputs);

		for (int i = 0; i < neurons.size(); i++) {
			neurons[i].activatedOutput = softmaxVals[i];
		}
	}
	else {
		std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
			n.forward(inputs);
		});
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
	std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
		n.accumulateGradients(inputs);
	});
}
