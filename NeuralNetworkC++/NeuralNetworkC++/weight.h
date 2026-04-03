class weight {
	weight(int value);

	void resetGradient();

	int value;
	float gradient = 0.0f;
	float prev_update = 0.0f;
};