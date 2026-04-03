#include <string>
#include <list>
#include "weight.h"

class neuron {
	public:
		neuron(int numInputs, std::string activatationFunction);

		int forward(std::list<int> inputs);
		void accumulateGradients(std::list<int> inputs);
		void updateWeights(int learnRate);

		std::list<weight> weights;
		float bias = 0.0f;
		float rawOutput = 0.0f;
		float activatedOutput = 0.0f;
		float delta = 0.0f;
};