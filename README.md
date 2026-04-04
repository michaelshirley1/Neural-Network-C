# Neural Network in C++

A feedforward neural network built from scratch in C++ — no ML libraries

---

## What it is

Fully connected neural network supporting:
- **Activations:** ReLU, Sigmoid, Tanh, Linear, Softmax
- **Loss functions:** Cross-entropy, MSE
- **Training:** Mini-batch SGD with backpropagation
- **Weight init:** Xavier initialisation
- **Input loading:** Image datasets via stb_image (folder/class/*.jpg)

---

## Architecture

```
Weight         — scalar value + gradient + momentum term
Neuron         — weights, bias, forward pass, delta, gradient accumulation
InputNeuron    — lightweight input node, no weights
Layer          — collection of neurons, handles Softmax as a group
Network        — full forward/backward/training loop
helpers        — activation functions, derivatives, loss, data loading
```

---

## Quickstart

```cpp
std::vector<layerConfig> configs = {
    { 1024, applyTypes::LINEAR,  networkLayerType::INPUT  },
    {  256, applyTypes::RELU,    networkLayerType::HIDDEN },
    {   39, applyTypes::SOFTMAX, networkLayerType::OUTPUT },
};

network net(configs, 0.01f, lossType::CROSS_ENTROPY);

net.trainBatch(inputs, actuals);

std::vector<float> prediction = net.forward(input);
```

---

## Training on images

Drop your dataset in, Load it with the input helper:

```cpp
input loader;
std::vector<std::vector<float>> inputs, actuals;
loader.loadDataset("path/to/data", labels, inputs, actuals);
```
Currently only supports images, adding support for any types in future releases.
Images are loaded as grayscale, flattened to 1D, and normalised to `0.0–1.0`.

---

## Project structure

```
helpers/        helper functions, activation functions, loss, data loading, types
layer/          layer forward/backward
network/        full network, training loop
neuron/         neuron forward/backward, weight update
weight/         weight struct, gradient reset
main.cpp        entry point
```

---

## Built with

- C++17
- stb_image (single-header image loading)
