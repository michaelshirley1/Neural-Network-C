#include "network.h"
#include <math-helper/helpers.h>
#include <fstream>
#include <stdexcept>
#include <execution>

network::network(const std::vector<layerConfig>& configs, float initLearningRate, lossType initLossFn) {
	learningRate = initLearningRate;
	lossFn = initLossFn;

	for (int i = 0; i < configs.size(); i++) {
		int numInputs = (i == 0) ? 0 : configs[i - 1].size;

		layers.push_back(layer(
			configs[i].size,
			numInputs,
			configs[i].activation,
			configs[i].layerType
		));
	}
}

std::vector<float> network::forward(const std::vector<float>& inputs) {
	std::vector<float> current = inputs;
    for (auto& l : layers) {
		current = l.forward(current);
	}
	return current;
}

void network::backward(const std::vector<float>& prediction, const std::vector<float>& actual) {
	std::vector<float> errors = helpers::lossDerivative(lossFn, prediction, actual);

	for (int i = layers.size() - 1; i >= 1; i--) {
		errors = layers[i].backward(errors);
	}
}

void network::accumulateGradients() {
	for (int i = 1; i < layers.size(); i++) {
		std::vector<float> inputs = layers[i - 1].getOutputs();
		layers[i].accumulateGradients(inputs);
	}
}

void network::updateWeights() {
	for (int i = 1; i < layers.size(); i++) {
		layers[i].updateWeights(learningRate);
	}
}

float network::trainOneSample(const std::vector<float>& inputData, const std::vector<float>& actual) {
	std::vector<float> prediction = forward(inputData);
	float loss = helpers::computeLoss(lossFn, prediction, actual);
	backward(prediction, actual);
	accumulateGradients();
	return loss;
}

void network::saveWeights(const std::string& path) {
	std::ofstream file(path, std::ios::binary);
	if (!file) throw std::runtime_error("Cannot open file for writing: " + path);

	for (int i = 1; i < (int)layers.size(); i++) {
        std::for_each(std::execution::par_unseq, layers[i].neurons.begin(), layers[i].neurons.end(), [&](neuron& n) {
            std::for_each(std::execution::par_unseq, n.weights.begin(), n.weights.end(), [&](weight& w) {
				file.write(reinterpret_cast<const char*>(&w.value), sizeof(float));
			});

			file.write(reinterpret_cast<const char*>(&n.bias.value), sizeof(float));
		});
	}
}

void network::loadWeights(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) throw std::runtime_error("Cannot open file for reading: " + path);

	for (int i = 1; i < (int)layers.size(); i++) {
        std::vector<neuron> neurons = layers[i].neurons;
        std::for_each(std::execution::par_unseq, neurons.begin(), neurons.end(), [&](neuron& n) {
            std::for_each(std::execution::par_unseq, n.weights.begin(), n.weights.end(), [&](weight& w) {
				file.read(reinterpret_cast<char*>(&w.value), sizeof(float));
			});
			file.read(reinterpret_cast<char*>(&n.bias.value), sizeof(float));
		});
	}
}

float network::trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& actuals) {
	const int batchSize = 32;
	float totalLoss = 0.0f;
	for (int i = 0; i < (int)inputs.size(); i++) {
		totalLoss += trainOneSample(inputs[i], actuals[i]);
		if ((i + 1) % batchSize == 0 || i == (int)inputs.size() - 1) {
			updateWeights();
		}
	}
	return totalLoss / inputs.size();
}
