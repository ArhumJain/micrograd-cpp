#pragma once

#include "value.h"
#include "neuron.h"
#include "layer.h"
#include <vector>

namespace grad {
    
    enum Activation {
        TANH,
        RELU,
        SIGMOID,
        SOFTMAX,
    };

    class MLP {
    private:
        int inputCount;
    public:
        std::vector<Layer> layers;
        
        MLP();
        MLP(int numberOfInputs, std::initializer_list<int> layerSizes, Activation a = Activation::TANH);

        std::vector<Value> passInputs(std::vector<Value> inputs);
        std::vector<Value> parameters();
        void updateParameters(double learningRate);
        void zeroGrad();
    };
    
    class Loss {
    public:
        static Value meanSquaredError(std::vector<Value>& pred, std::vector<Value>& truth);
    };
}