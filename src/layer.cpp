#include "value.h"
#include "neuron.h"
#include "layer.h"
#include <vector>
 
namespace grad {

    Layer::Layer(int neuronInputCount, int neuronCount) {
        this->inputCount = neuronInputCount;
        this->neurons = std::vector<grad::Neuron>(neuronCount);
        std::cout << neuronInputCount << " " << neuronCount << std::endl;
        for (int i=0; i<neuronCount; i++) {
            neurons[i] = Neuron(neuronInputCount);
        }
    }

    std::vector<Value> Layer::passInputs(std::vector<Value> inputs) {
        if (inputs.size() !=  this->inputCount) {
            throw std::length_error("Input size does not match expected input size for Layer: Expected size " + std::to_string(this->inputCount));
        }
        std::vector<Value> outputs(this->neurons.size());
        // std::cout << "Layer Expect Input Size is: " << this->inputCount << ", Layer Expected Output Size is: " << this->neurons.size() << std::endl;
        // std::cout << inputs.size() << std::endl;
        for (int i=0; i<this->neurons.size(); i++) {
            outputs[i] = this->neurons[i].passInputs(inputs);
        }

        return outputs;
    }
    
    std::vector<Value> Layer::parameters() {
        std::vector<Value> p;
        std::vector<Value> addParameters;
        for (auto& n: this->neurons) {
            addParameters = n.parameters();
            p.insert(p.end(), addParameters.begin(), addParameters.end());
        }
        return p;
    }
    
}