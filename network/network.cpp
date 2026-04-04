#include "network.h"
#include "helpers.h"

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

float network::trainBatch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& actuals) {
	float totalLoss = 0.0f;
	for (int i = 0; i < inputs.size(); i++) {
		totalLoss += trainOneSample(inputs[i], actuals[i]);
	}
	updateWeights();
	return totalLoss / inputs.size();
}
