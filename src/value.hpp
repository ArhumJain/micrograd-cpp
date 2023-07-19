
#pragma once
#include "value_network.hpp"

#include <iostream>
#include <string.h>
#include <math.h>

#include <random>
#include <fstream>
#include <cstring>
#include <ctime>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>

class Value 
{
    
    friend class ValueNetwork;
    friend Value operator+(const int &value, Value &other);
    friend Value operator-(const int &value, Value &other);
    friend Value operator*(const int &value, Value &other);

private:
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
    
    void deepCopy(Value* current, Value* other) {
        current->data = other->data;
        current->grad = other->grad;
        current->label = other->label;
        current->op = other->op;

        if (other->prev[0] == nullptr) {
            return;
        }

        current->prev[0] = new Value();
        deepCopy(current->prev[0], other->prev[0]);
        if (other->prev[1] != nullptr) {
            current->prev[1] = new Value();
            deepCopy(current->prev[1], other->prev[1]);
        }
    }
    
    Value(Value &value, bool check) {
        deepCopy(this, &value);
    }
public:
    Value* prev[2] = {nullptr, nullptr};
    double data; 
    double grad = 0.0;
    std::string label = "";
    Value() {
        this->data = 0.0;
        this->op = "";
    }
    Value(double data, std::initializer_list<Value> children = {}, std::string op = "") {
        this->data = data;
        int i = 0;
        for (auto child: children) {

            prev[i] = new Value(child, true);
            i++;
        }
        this->op = op;
    }
    
    friend std::ostream & operator << (std::ostream &out, Value &v) {
        out << "Value(data=" << v.data << ")" << std::endl;
        return out;
    }
    
    Value operator=(const double &value) {
        return Value(value);
    }

    Value operator+(const double &value) {
        Value v = Value(value);
        return (*this) + v;
    }
    Value operator+(Value &value) {
        double sum = this->data + value.data;
        Value current = *this;
        Value v = Value(sum, {current, value}, "+");
        return v;
    }

    Value operator-(Value &value) {
        double difference = this->data - value.data;
        Value current = *this;
        Value v = Value(difference, {current, value}, "-");
        return v;
    }

    Value operator*(Value &value) {
        double product = this->data * value.data;
        Value current = *this;
        Value v = Value(product, {current, value}, "*");
        return v;
    }
    
    Value tanh() {
        double x = this->data;
        double t = (std::exp(x * 2)-1)/(std::exp(x * 2)+1);

        Value current = *this;
        Value out = Value(t, {current}, "tanh");
        return out;

    }

    Value exp() {
        
    }

    void backpropagate() {
        this->grad = 1.0;
        this->internalBackpropagate();
    }
};

Value operator+(const int &value, Value &other) {
    Value v = Value(value);
    return v + other;
}
Value operator-(const int &value, Value &other) {
    Value v = Value(value);
    return v - other;
}
Value operator*(const int &value, Value &other) {
    Value v = Value(value);
    return v * other;
}