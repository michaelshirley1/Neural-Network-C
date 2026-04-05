#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "network.h"
#include "input-helper/input.h"

int main() {
    std::cout << "Starting network" << std::endl;

    const std::string fileLocation = "";

    std::vector<std::string> labels = std::vector<std::string> {
        "Z",
        "Y",
        "X",
        "W",
        "V",
        "U",
        "T",
        "S",
        "R",
        "Q",
        "P",
        "N",
        "M",
        "L",
        "K",
        "J",
        "I",
        "H",
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
        "@",
        "9",
        "8",
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
        "1",
        "0",
        "&",
        "$",
        "#"
    };

    std::vector<std::vector<float>> inputs = std::vector<std::vector<float>>();
    std::vector<std::vector<float>> actuals = std::vector<std::vector<float>>();

    input::loadDataset(fileLocation, labels, inputs, actuals);

    std::vector<int> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<std::vector<float>> shuffledInputs(inputs.size());
    std::vector<std::vector<float>> shuffledActuals(actuals.size());
    for (int i = 0; i < (int)indices.size(); i++) {
        shuffledInputs[i] = inputs[indices[i]];
        shuffledActuals[i] = actuals[indices[i]];
    }
    inputs = std::move(shuffledInputs);
    actuals = std::move(shuffledActuals);

    std::vector<layerConfig> configs = {
        { 1024, applyTypes::LINEAR,  networkLayerType::INPUT  },
        { 256, applyTypes::RELU,    networkLayerType::HIDDEN },
        { 39, applyTypes::SOFTMAX, networkLayerType::OUTPUT },
    };

    network net(configs, 0.01f, lossType::CROSS_ENTROPY);

    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = net.trainBatch(inputs, actuals);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " — Loss: " << loss << std::endl;
        }
    }

    net.saveWeights("weights.bin");
    std::cout << "Weights saved to weights.bin\n";

    std::cout << "\nPredictions after training:\n";
    for (int i = 0; i < inputs.size(); i++) {
        std::vector<float> pred = net.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << ", " << inputs[i][2] << "]"
                  << "  =>  [" << pred[0] << ", " << pred[1] << "]"
                  << "  (expected [" << actuals[i][0] << ", " << actuals[i][1] << "])\n";
    }

    return 0;
}
