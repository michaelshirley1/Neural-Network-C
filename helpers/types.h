#pragma once

enum class networkLayerType {
	INPUT = 1,
	HIDDEN = 2,
	OUTPUT = 3,
};

enum class lossType {
	MSE = 1,
	CROSS_ENTROPY = 2,
};

enum class applyTypes {
	RELU = 1,
	SIGMOID = 2,
	TANH = 3,
	LINEAR = 4,
	SOFTMAX = 5,
};
