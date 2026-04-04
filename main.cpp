#include <iostream>
#include <vector>
#include "network.h"

#define fileLocation = "C:\Users\micha\Downloads\archive"

int main() {
    


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

    std::cout << "\nPredictions after training:\n";
    for (int i = 0; i < inputs.size(); i++) {
        std::vector<float> pred = net.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << ", " << inputs[i][2] << "]"
                  << "  =>  [" << pred[0] << ", " << pred[1] << "]"
                  << "  (expected [" << actuals[i][0] << ", " << actuals[i][1] << "])\n";
    }

    return 0;
}
