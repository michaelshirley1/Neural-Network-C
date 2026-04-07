#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "network.h"
#include "input-helper/input.h"

/*
Welcome to the annotated version of this repository, neural networks can be quite difficult to understand. I hope to break it down in such a way
that anyone can read and understand this. Even without prior computer science education.

What is a Neural Network?

Simply, a neural network can be described as, a series of "weights" and "biases" which can take in an input, and produce an output.
Lets say I have an image

 __         __
/  \.-"""-./  \
\    -   -    /
 |   o   o   |
 \  .-'''-.  /
  '-\__Y__/-'
     `---`

This image in digital space is made up of a series of pixel, each pixel has a RBG value. Lets say I take this image and cut it up into its 
individual pixels. Heres an example of the top of ears of this happy bear ->  [, ,_,_, , , , , , , , ,_,_, ]
Since this image is in black and white we can represent this as 1s and 0s, the pixel can be ON or OFF. 

[ , ,_,_, , , , , , , , ,_,_, ]
[0,0,1,1,0,0,0,0,0,0,0,0,1,1,0]

Now lets imagine we would like to which bears are happy bears and which are sad bears. If a bear is sad they probably have their ears down right?

  __.-"""-.__  
 /  -   -    \
 |   _   _   |
 \  .-'''-.  /
  '-\__Y__/-'
     `---`

Look how sad this bear is. His ears have basically fallen off. I would be sad too. 
Lets read the top of this bears head again.

[ , , , , ,",",", , , , , , , ]
[0,0,0,0,0,1,1,1,0,0,0,0,1,1,0]

Nice! we can see that if this bear is sad, the position of where the 0s and 1s are are different right? Now, how does this relate to neural networks?

                (H1)
               / | \ \
   (I1)------/  |  \  \
     |  \   /   |   \  \
     |   \ /    |    \  \
     |    X     |     \  \
     |   / \    |      \  \
     |  /   \   |       \  \
   (I2)-------(H2)-------(O1)
     |  \   /   |       /  /
     |   \ /    |      /  /
     |    X     |     /  /
     |   / \    |    /  /
     |  /   \   |   /  /
   (I3)------\  |  /  /
               \ | / /
                (H3)

Here is a diagram of a neural network, you have your input, your hidden bias, and your output. In our case we are trying to figure out which bears
are sad and which bears are not.

Lets say an image of a bear is 8 x 8 pixels, if we take every single pixel and put them in a long list of 0s and 1s, it would be 64 pixels long.
Since we are trying to decode these images, we would have 64 nodes on the input side. One for each pixel of our image. Each of the input nodes
have a connection to every single one of the hidden nodes, the math behind it is not relevant now, but just try to imagine every single connection.
Those hidden nodes then have a connection to the output notes, for this example we will have 2 output notes. Happy bear, Sad bear.

Its training time for this neural network! 

We start by feeding the network its first image. Let's say it's a happy bear. 
The 64 pixel values flow into the 64 input nodes, signals travel through every connection to the hidden 
nodes, and finally the output nodes light up with two numbers. Maybe it says: Happy bear: 40%, Sad bear: 60%.

This is where the magic happens.
The network looks at how wrong it was this wrongness is called the "loss"
It then works backwards through every single connection, asking: "Did this connection help or hurt the answer?" 
Connections that pushed toward the wrong answer get slightly weakened

How slightly? That's controlled by the learning rate
Now we feed it the next image. A sad bear. Again it guesses. Again it checks how wrong it was. Again it nudges. 
Then the next image. And the next. Hundreds, thousands of times.

At first the network is basically guessing, after thousands and thousands of retrys it can slowly nudge closer and closer to the correct answer.

Its not actually magic (that would be cool tho huh), just math. 

Lets go back to the first example. The sad bear. How floppy its ears are. lets say the ears are pixels 12 and 16 in our list of pixels. Every sad bear
will have a difference in pixels 12 and 16 compared to their happy counterparts. So every single connection thats hitting the 12th and 16th pixel will have slightly different weights compared to a happy bear. Those weights are connected
to the output so if we pass through that image, a 1 in pixel 12 or 16 will work its way down each connection and be multiplied by each weight on a node and hit the output as a percentage of likelyhood that theyre sad or happy.
*/


/* These are all of my outputs and their corresponding labels */
static const std::vector<std::string> LABELS = {
    "Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q",
    "P", "N", "M", "L", "K", "J", "I", "H", "G", "F",
    "E", "D", "C", "B", "A", "@",
    "9", "8", "7", "6", "5", "4", "3", "2", "1", "0",
    "&", "$", "#"
};

/* Because the images I have are 32 x 32 I have 1024 input notes */
/* The number of hidden nodes is usually proportonal to the amount of input notes */
/* I have 39 total labels, so I have 39 total outputs */
static const std::vector<layerConfig> CONFIGS = {
    { 1024, applyTypes::LINEAR,  networkLayerType::INPUT  },
    { 256,  applyTypes::RELU,    networkLayerType::HIDDEN },
    { 39,   applyTypes::SOFTMAX, networkLayerType::OUTPUT },
};

/* 
runPredict is only run whenever I have an already trained neural network 
I set up the same network as I trained the weights on
Load the weights
Load the image I would like to predict the output of
Pass it through my neural network
Then state the predicted output
No neural network is 100% accurate, so we use a series of percentages of likelyhood to represent the best answer
S: 75%
5: 22%
G: 1%
etc
*/
void runPredict(const std::string& imagePath) {
    /* This is explained in the training function */
    network net(CONFIGS, 0.0001f, lossType::CROSS_ENTROPY);
    net.loadWeights("weights.bin");

    std::vector<float> imageData = input::loadImage(imagePath);
    /* The image is loaded, lets push it through our neural network using the forward command */
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

/*
This is where the magic happens, the training script
I start by initalising my list of inputs (inputs) and their answer (actuals)
I then run through every single image in my training data and 
*/
void runTrain(const std::string& datasetPath) {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> actuals;

    input::loadDataset(datasetPath, LABELS, inputs, actuals);

    /* Due to errors that can arrise from ordering your data in groupings of output, we need to shuffle them to get accurate training */
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

    /*
    What does this mean?

    We are starting up an instance of our neural network
    The 3 parameters, 
    CONFIGS:        Our input, hidden and output, how many neurons per and what function we are using for each

    learningRate:   learning rate controls how much the model adjusts its weights after each training step, During training, a neural network calculates its error (loss) and uses gradient descent to nudge weights in the direction that reduces that error. The learning rate scales how big that
                    nudge is
    
    lossType:       this is the algorithm that measures how wrong our models probabilities are compared to the actual answer. Cross entropy is explained in types.h
    */
    network net(CONFIGS, 0.0001f, lossType::CROSS_ENTROPY);

    int epochs = 7;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = net.trainBatch(inputs, actuals);
        std::cout << "Epoch " << epoch << " - Loss: " << loss << std::endl;
    }

    net.saveWeights("weights.bin");
    std::cout << "Weights saved to weights.bin\n";
}


/*
This is the "main" function, the thing that runs the show
For simplicity of use I run a few console commands which allow you select between train or predict,
you can then link a repo of images you have and train from that, or specify an image you would like to predict the output of
*/
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
