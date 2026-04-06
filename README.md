# Neural Network in C++

A feedforward neural network built from scratch in C++, no added ML libraries, all done by hand

This repository serves as a easy to read machine learning example for others trying to get into Machine learning. No obfuscation of the mathematics used, no external libraries.

**A annotated version is avaliable in /annotated**

<img width="661" height="261" alt="image" src="https://github.com/user-attachments/assets/0b999a45-c8d9-4172-b99b-4f71fec53c4c" />

## What it is

Fully connected neural network supporting:
- Activations: ReLU, Sigmoid, Tanh, Linear, Softmax
- Loss functions: Cross-entropy, MSE
- Training: Mini-batch SGD with backpropagation
- Weight init: Xavier initialisation
- Input loading: Image datasets via stb_image (folder/class/*.jpg)

## Architecture

```
Weight         — scalar value + gradient + momentum term
Neuron         — weights, bias, forward pass, delta, gradient accumulation
Layer          — collection of neurons, handles Softmax as a group
Network        — full forward/backward/training loop
```

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
