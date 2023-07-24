#include "value.h"
#include "neuron.h"
#include "utils.h"
#include <random>

namespace grad {

    Neuron::Neuron() {}

    Neuron::Neuron(int inputCount) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        this->weights = std::vector<Value>(inputCount);
        for (int i=0; i < inputCount; i++) {
            this->weights[i] = Value(dist(Utils::generator));
        }
        
        this->bias = dist(Utils::generator);
    }

    Value Neuron::passInputs(std::vector<Value> inputs) {
        if (inputs.size() != this->weights.size()) {
            throw std::length_error("Input size does not match expected size of Neuron weights: Expected size " + std::to_string(this->weights.size()));
        }

        Value dot = this->bias.data();
        for (int i=0; i<this->weights.size(); i++) {
            dot = dot + inputs[i] * weights[i];
        }

        // Value d = dot.tanh();
        return dot.tanh();
    }

    std::vector<Value> Neuron::parameters() {
        std::vector<Value> p = weights;
        p.push_back(bias);
        return p;
    }
}