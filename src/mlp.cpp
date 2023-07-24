#include "layer.h"
#include "neuron.h"
#include "value.h"
#include "mlp.h"

namespace grad {

    MLP::MLP() {}

    MLP::MLP(int numberOfInputs, std::initializer_list<int> layerSizes, Activation a /*Activatino::TANH*/) {
        this->inputCount = numberOfInputs;
        int prev = this->inputCount;
        for (auto& i: layerSizes) {
            layers.push_back(Layer(prev, i));
            prev = i;
        }
    }

    std::vector<Value> MLP::passInputs(std::vector<Value> inputs) {
        std::vector<Value> outputs = inputs;
        int whichLayer = 0;      
        for (Layer& layer: this->layers) {
            // std::cout << "HI" << std::endl;
            // std::cout << outputs.size() << std::endl;
            // if (outputs[0].data() != outputs[0].data()) {
            //     std::cout << "IS NAN WHY!!" << std::endl;
            //     std::cout << layer.neurons.size() << std::endl;
            //     std::cout << whichLayer << std::endl;
            //     return {};
            // }
            outputs = layer.passInputs(outputs);
            whichLayer++;
            // std::cout << outputs.size() << std::endl;
        }
        
        return outputs;
    }
    
    std::vector<Value> MLP::parameters() {
        std::vector<Value> p;
        std::vector<Value> addParameters;
        for (auto& l: this->layers) {
            addParameters = l.parameters();
            p.insert(p.end(), addParameters.begin(), addParameters.end());
        }
        return p;
    }

    void MLP::updateParameters(double learningRate) {
        std::vector<Value> params = this->parameters();

        // if (params.size() != gradient.size()) {
        //     throw std::length_error("Gradient vector size does not match expected paramter vector size. Expected size: " + std::to_string(params.size()));
        // }
        for (int i=0; i<params.size(); i++) {
            Value& param = params[i];
            param.setData(param.data() + learningRate * param.grad());
        }

    }

    void MLP::zeroGrad() {
        std::vector<Value> params = this->parameters();
        for (auto& p: params) {
            p.setGrad(0);
        }
    }
    
    Value Loss::meanSquaredError(std::vector<Value>& pred, std::vector<Value>& truth) {
        if (pred.size() != truth.size()) {
            throw std::length_error("Prediction vector size does not match expected truth vector size");
        }
        
        Value loss = 0;
        for (int i=0; i<pred.size(); i++) {
            loss = loss + (pred[i]-truth[i]).raiseTo(2);
        }

        return loss;
    }
}