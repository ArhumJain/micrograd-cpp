#pragma once

#include "value.h"
#include "neuron.h"
#include <vector>

namespace grad {
    
    class Layer {
    private:
        int inputCount;
    public:
        std::vector<Neuron> neurons;
        
        Layer(int neuronInputCount, int neuronCount);

        std::vector<Value> passInputs(std::vector<Value> inputs);
        std::vector<Value> parameters();
    };
}