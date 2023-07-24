#pragma once

#include "value.h"
#include <vector>

namespace grad {
    
    class Neuron {

    public:
        std::vector<Value> weights;
        Value bias;
        
        Neuron();
        Neuron(int inputCount);

        Value passInputs(std::vector<Value> inputs);
        std::vector<Value> parameters();
    };
}