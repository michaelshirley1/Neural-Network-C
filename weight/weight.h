#pragma once

class weight {
public:
	weight(float value);

	void resetGradient();
	weight& operator-=(float rhs);

	float value;
	float gradient = 0.0f;
	float prev_update = 0.0f;
};
