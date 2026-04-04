#pragma once
#include "types.h"

class layerConfig {
public:
	layerConfig(int initSize, networkLayerType initNetworkLayer, applyTypes initApplyType) {
		size = initSize;
		networkLayer = initNetworkLayer;
		applyType = initApplyType;
	}

	networkLayerType networkLayer;
	applyTypes applyType;
	int size;
};