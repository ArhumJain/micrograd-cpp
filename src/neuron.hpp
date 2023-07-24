#pragma once
#include <vector>
#include <random>
#include <ctime>
#include "value.hpp"
#include "value_network.hpp"
#include "utils.hpp"
#include <iostream>

namespace grad {

enum Activation {
    TANH,
    RELU,
    SIGMOID,
    SOFTMAX,
};

class Neuron {

private:
public:
    std::vector<Value> weights;
    Value bias;

    Neuron() {};
    
    Neuron(int inputCount) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        this->weights = std::vector<Value>(inputCount);
        for (int i=0; i < inputCount; i++) {
            // this->weights[i] = Value(dist(Utils::generator));
            this->weights[i] = 0.5;
        }
        
        // this->bias = dist(Utils::generator);
        this->bias = 0.5;
    }

    Value passInputs(std::vector<Value> inputs) {
        if (inputs.size() != this->weights.size()) {
            throw std::length_error("Input size does not match expected size of Neuron weights: Expected size " + std::to_string(this->weights.size()));
        }

        Value dot = this->bias.data();
        for (int i=0; i<this->weights.size(); i++) {
            dot = dot + inputs[i] * weights[i];
        }

        return dot.tanh();
    }

    std::vector<Value> parameters() {
        std::vector<Value> p = weights;
        p.push_back(bias);
        return p;
    }

};

class Layer {
    
private:
    int inputCount;
public:
    std::vector<Neuron> neurons;
    
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

        return outputs;
    }
    
    std::vector<Value> parameters() {
        std::vector<Value> p;
        std::vector<Value> addParameters;
        for (auto& n: this->neurons) {
            addParameters = n.parameters();
            p.insert(p.end(), addParameters.begin(), addParameters.end());
        }
        return p;
    }
    
};

class MLP {

private:
    int inputCount;
public:
    std::vector<Layer> layers;
    
    MLP() {}

    MLP(int numberOfInputs, std::initializer_list<int> layerSizes, Activation a = Activation::TANH) {
            this->inputCount = numberOfInputs;
        int prev = this->inputCount;
        for (auto& i: layerSizes) {
            layers.push_back(Layer(prev, i));
            prev = i;
        }
    }

    std::vector<Value> passInputs(std::vector<Value> inputs) {
        std::vector<Value> outputs = inputs;       
        for (Layer& layer: this->layers) {
            // std::cout << "HI" << std::endl;
            // std::cout << outputs.size() << std::endl;
            outputs = layer.passInputs(outputs);
            // std::cout << outputs.size() << std::endl;
        }
        
        return outputs;
    }
    
    std::vector<Value> parameters() {
        std::vector<Value> p;
        std::vector<Value> addParameters;
        for (auto& l: this->layers) {
            addParameters = l.parameters();
            p.insert(p.end(), addParameters.begin(), addParameters.end());
        }
        return p;
    }

    void updateParameters(double learningRate) {
        std::vector<Value> params = this->parameters();

        // if (params.size() != gradient.size()) {
        //     throw std::length_error("Gradient vector size does not match expected paramter vector size. Expected size: " + std::to_string(params.size()));
        // }
        for (int i=0; i<params.size(); i++) {
            Value& param = params[i];
            param.setData(param.data() + learningRate * param.grad());
        }

    }

    void zeroGrad() {
        std::vector<Value> params = this->parameters();
        for (auto& p: params) {
            p.setGrad(0);
        }
    }
};

class Loss {

public:
    static Value meanSquaredError(std::vector<Value>& pred, std::vector<Value>& truth) {
        if (pred.size() != truth.size()) {
            throw std::length_error("Prediction vector size does not match expected truth vector size");
        }
        
        Value loss = 0;
        for (int i=0; i<pred.size(); i++) {
            loss = loss + (pred[i]-truth[i]).raiseTo(2);
        }

        return loss;
    }

};

}