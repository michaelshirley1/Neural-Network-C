#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "network.h"
#include "input-helper/input.h"

static const std::vector<std::string> LABELS = {
    "Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q",
    "P", "N", "M", "L", "K", "J", "I", "H", "G", "F",
    "E", "D", "C", "B", "A", "@",
    "9", "8", "7", "6", "5", "4", "3", "2", "1", "0",
    "&", "$", "#"
};

static const std::vector<layerConfig> CONFIGS = {
    { 1024, applyTypes::LINEAR,  networkLayerType::INPUT  },
    { 256,  applyTypes::RELU,    networkLayerType::HIDDEN },
    { 39,   applyTypes::SOFTMAX, networkLayerType::OUTPUT },
};

void runPredict(const std::string& imagePath) {
    network net(CONFIGS, 0.0001f, lossType::CROSS_ENTROPY);
    net.loadWeights("weights.bin");

    std::vector<float> imageData = input::loadImage(imagePath);
    std::vector<float> output = net.forward(imageData);

    std::vector<int> idx(output.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
        [&](int a, int b) { return output[a] > output[b]; });

    std::cout << "\n=== Prediction ===\n";
    std::cout << "Letter: " << LABELS[idx[0]] << "\n";
    std::cout << "Confidence: " << output[idx[0]] * 100.0f << "%\n";
    std::cout << "\nTop 5:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  " << LABELS[idx[i]] << "  " << output[idx[i]] * 100.0f << "%\n";
    }
}

void runTrain(const std::string& datasetPath) {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> actuals;

    input::loadDataset(datasetPath, LABELS, inputs, actuals);

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

    network net(CONFIGS, 0.01f, lossType::CROSS_ENTROPY);

    int epochs = 3;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = net.trainBatch(inputs, actuals);
        std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
    }

    net.saveWeights("weights.bin");
    std::cout << "Weights saved to weights.bin\n";
}

int main(int argc, char* argv[]) {
    std::string mode, path;

    if (argc >= 3) {
        mode = argv[1];
        path = argv[2];
    } else {
        std::cout << "Choose mode (train / run): ";
        std::cin >> mode;

        if (mode == "train") {
            std::cout << "Enter dataset folder path: ";
        } else if (mode == "run") {
            std::cout << "Enter image path: ";
        } else {
            std::cerr << "Unknown mode '" << mode << "'. Use 'train' or 'run'.\n";
            return 1;
        }

        std::cin >> path;
    }

    if (mode == "train") {
        runTrain(path);
    } else if (mode == "run") {
        runPredict(path);
    } else {
        std::cerr << "Unknown mode '" << mode << "'. Use 'train' or 'run'.\n";
        return 1;
    }

    return 0;
}
