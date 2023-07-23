#pragma once
#include <vector>
#include <random>
#include <ctime>
#include "value.hpp"
#include "value_network.hpp"
#include "utils.hpp"
#include <iostream>

namespace grad {

class Neuron {

private:
    std::vector<Value> weights;
    Value bias;
public:
    Neuron() {};
    
    Neuron(int inputCount) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        this->weights = std::vector<Value>(inputCount);
        for (int i=0; i < inputCount; i++) {
            this->weights[i] = Value(dist(Utils::generator));
        }
        
        this->bias = dist(Utils::generator);
    }

    Value passInputs(std::vector<Value> inputs) {
        if (inputs.size() != this->weights.size()) {
            throw std::length_error("Input size does not match expected size of Neuron weights: Expected size " + std::to_string(this->weights.size()));
        }

        Value dot = this->bias.data();
        for (int i=0; i<this->weights.size(); i++) {
            std::cout << inputs[i].pimpl.use_count() << std::endl;
            dot = dot + inputs[i] * weights[i];
            std::cout << inputs[i].pimpl.use_count() << std::endl;
        }

        return dot.tanh();
    }

};

class Layer {
    
private:
    std::vector<Neuron> neurons;
    int inputCount;
public:
    Layer(int neuronInputCount, int neuronCount) {
        this->inputCount = neuronInputCount;
        this->neurons = std::vector<grad::Neuron>(neuronCount);
        for (int i=0; i<neuronCount; i++) {
            neurons[i] = Neuron(neuronInputCount);
        }
    }

    std::vector<Value> passInputs(std::vector<Value> inputs) {
        if (inputs.size() !=  this->inputCount) {
            throw std::length_error("Input size does not match expected input size for Layer: Expected size " + std::to_string(this->inputCount));
        }

        std::vector<Value> outputs(this->neurons.size());
        // std::cout << "Layer Expect Input Size is: " << this->inputCount << ", Layer Expected Output Size is: " << this->neurons.size() << std::endl;
        // std::cout << inputs.size() << std::endl;
        for (int i=0; i<this->neurons.size(); i++) {
            outputs[i] = this->neurons[i].passInputs(inputs);
        }


        if (neurons.size() == 1) {
            char filePath[] = "./test.png";
            ValueNetwork test(outputs[0]);
            test.createGraph(filePath);
        }

        return outputs;
    }
};

class MLP {

private:
    std::vector<Layer> layers;
    int inputCount;
public:
    MLP() {}

    MLP(int numberOfInputs, std::initializer_list<int> layerSizes) {
            this->inputCount = numberOfInputs;
        int prev = this->inputCount;
        for (auto& i: layerSizes) {
            layers.push_back(Layer(prev, i));
            prev = i;
        }
    }

    std::vector<Value> passInputs(std::vector<Value>& inputs) {
        std::vector<Value>& outputs = inputs;       
        for (Layer& layer: this->layers) {
            // std::cout << "HI" << std::endl;
            // std::cout << outputs.size() << std::endl;
            outputs = layer.passInputs(outputs);
            // std::cout << outputs.size() << std::endl;
        }
        
        return outputs;
    }
};

}