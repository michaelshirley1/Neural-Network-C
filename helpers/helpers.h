#pragma once
#include <vector>
#include "types.h"

class helpers {
public:
	static float randomUniform(float low, float high);
	static float apply(applyTypes activationFunction, float value);

	static float relu(float value);
	static float sigmoid(float value);
	static float tanh(float value);
	static float linear(float value);
	static std::vector<float> softmax(std::vector<float> valueVector);

	static float activationDerivative(applyTypes activationFunction, float z);
	static float reluDerivative(float z);
	static float sigmoidDerivative(float z);
	static float tanhDerivative(float z);
	static float linearDerivative(float z);
};
