#pragma once

#include <iostream>
#include <string.h>
#include <math.h>
#include "value_network.hpp"

#include <random>
#include <fstream>
#include <cstring>
#include <ctime>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>

class Value 
{
    
    friend class ValueNetwork;

private:
    Value* prev[2] = {nullptr, nullptr};
    std::string op;

    void backward() {
        if (this->op == "+") {
            prev[0]->grad += this->grad;
            prev[1]->grad += this->grad;
        } else if (this->op == "*") {
            prev[0]->grad += prev[1]->data * this->grad;
            prev[1]->grad += prev[0]->data * this->grad;
        } else if (this->op == "tanh") {
            prev[0]->grad += (1 - (this->data * this->data)) * this->grad;
        }
    }

    void internalBackpropagate() {
        if (this->op == "") return;
        this->backward();
        for (int i=0; i<2; i++) {
            Value* child = prev[i];
            if (child != nullptr) {
                child->internalBackpropagate();
            }
        }
    }
public:
    double data; 
    double grad = 0.0;
    std::string label = "";
    Value(double data, std::initializer_list<Value*> children = {}, std::string op = "") {
        this->data = data;
        int i = 0;
        for (auto child: children) {
            prev[i] = child;
            i++;
        }
        this->op = op;
    }
    
    friend std::ostream & operator << (std::ostream &out, Value &v) {
        out << "Value(data=" << v.data << ")" << std::endl;
        return out;
    }
    
    Value operator+(Value &value) {
        double sum = this->data + value.data;
        Value v = Value(sum, {this, &value}, "+");
        return v;
    }

    Value operator-(Value &value) {
        double difference = this->data - value.data;
        Value v = Value(difference, {this, &value}, "-");
        return v;
    }

    Value operator*(Value &value) {
        double product = this->data * value.data;
        Value v = Value(product, {this, &value}, "*");
        return v;
    }
    
    Value tanh() {
        double x = this->data;
        double t = (std::exp(x * 2)-1)/(std::exp(x * 2)+1);

        Value out = Value(t, {this}, "tanh");
        return out;

    }

    void backpropagate() {
        this->grad = 1.0;
        this->internalBackpropagate();
    }
};