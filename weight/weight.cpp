#include "weight.h"

weight::weight(float initValue) {
	value = initValue;
	gradient = 0.0f;
	prev_update = 0.0f;
}

void weight::resetGradient() {
	weight::gradient = 0.0f;
}

weight& weight::operator-=(float rhs) {
	value -= rhs;
	return *this;
}
