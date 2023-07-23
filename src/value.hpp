
#pragma once

#include <iostream>
#include <string.h>
#include <math.h>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "value_network.hpp"

namespace grad 
{

class Value {
    friend class ValueNetwork;
private:
    class impl : public std::enable_shared_from_this<impl>
    {
    private:
        std::shared_ptr<impl> prev[2] = {nullptr, nullptr};
        // Value& outer; 

        void backward() {
            if (this->op == "+") {
                prev[0]->grad += 1.0 * this->grad;
                prev[1]->grad += 1.0 * this->grad;
            } else if (this->op == "-") {
                prev[0]->grad += this->grad;
                prev[1]->grad += -1.0 * this->grad;
            } else if (this->op == "*") {
                prev[0]->grad += prev[1]->data * this->grad;
                prev[1]->grad += prev[0]->data * this->grad;
            } else if (this->op == "tanh") {
                prev[0]->grad += (1 - (this->data * this->data)) * this->grad;
            } else if (this->op == "exp") {
                prev[0]->grad += this->data * this->grad;
            } else if (this->op == "^") {
                double exponent = prev[1]->data;
                prev[0]->grad += exponent * (std::pow(prev[0]->data, exponent-1)) * this->grad; 
            }
        }

        void sortIntoTopologicalOrder(std::shared_ptr<impl> &current, 
                                      std::unordered_set<Value::impl*> &explored,
                                      std::unordered_map<int, Value::impl*> &order, 
                                      int &currentIndex) {
            
            if (current->op != "") {
                for (int i=0; i<2; i++) {
                    std::shared_ptr<impl>& child = current->prev[i];
                    if (child != nullptr && explored.find(child.get()) == explored.end()) {
                        sortIntoTopologicalOrder(child, explored, order, currentIndex);
                    }
                }
            }

            explored.insert(current.get());
            order[currentIndex] = current.get();
            currentIndex++;
            return;
        }

    public:
        double data; 
        double grad = 0.0;
        std::string label = "";
        std::string op = "";
        
        impl(){
            this->data = 0.0;
        }
        
        impl(double data) {
            this->data = data;
        }

        impl(double data, std::string label) : impl(data) {
            // this->data = data;
            this->label = label;
        }

        impl(
            Value &outer, 
            double data, 
            const std::shared_ptr<impl>& a, 
            std::string op, 
            const std::shared_ptr<impl>& b = nullptr
            ) {

            this->data = data;
            prev[0] = std::shared_ptr<impl>(a);
            if (b != nullptr) {
                prev[1] = std::shared_ptr<impl>(b);
            }
            this->op = op;
        }

        std::shared_ptr<impl> getPtr() {
            return shared_from_this();
        }

        std::shared_ptr<impl>& getChild(const int &i) {
            return this->prev[i];
        }
        
        Value raiseTo(const double &value) {
            Value v = Value(value);
            return this->raiseTo(v);
        }
        
        Value raiseTo(Value &value) {
            double power = std::pow(this->data, value.data());
            Value v = Value(power, this->getPtr(), "^", value.pimpl);
            return v;
        }

        Value operator+(const double &value) {
            Value v = Value(value);
            return (*this) + v;
        }

        Value operator+(Value &&value) {
            double sum = this->data + value.data();
            Value v = Value(sum, this->getPtr(), "+", value.pimpl);
            return v;
        }

        Value operator+(Value &value) {
            double sum = this->data + value.data();
            Value v = Value(sum, this->getPtr(), "+", value.pimpl);
            return v;
        }

        Value operator-(const double &value) {
            Value v = Value(value);
            return (*this) - v;
        }

        Value operator-(Value &&value) {
            double sum = this->data - value.data();
            Value v = Value(sum, this->getPtr(), "-", value.pimpl);
            return v;
        }

        Value operator-(Value &value) {
            double sum = this->data - value.data();
            Value v = Value(sum, this->getPtr(), "-", value.pimpl);
            return v;
        }

        Value operator*(const double &value) {
            Value v = Value(value);
            return (*this) * v;
        }

        Value operator*(Value &&value) {
            double sum = this->data * value.data();
            Value v = Value(sum, this->getPtr(), "*", value.pimpl);
            return v;
        }

        Value operator*(Value &value) {
            double sum = this->data * value.data();
            Value v = Value(sum, this->getPtr(), "*", value.pimpl);
            return v;
        }

        Value operator/(const double &value) {
            Value v = Value(value);
            return (*this) / v;
        }

        Value operator/(Value &&value) {
            Value v = value.raiseTo(-1.0);
            return (*this) * v;
        }

        Value operator/(Value &value) {
            Value v = value.raiseTo(-1.0);
            return (*this) * v;
        }

        Value exp() {
            double x = this->data;
            // Value current = *this;
            Value v = Value(std::exp(x), this->getPtr(), "exp");
            return v;
        }

        Value tanh() {
            double x = this->data;
            double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);

            Value v = Value(t, this->getPtr(), "tanh");
            return v;
        }

        void backpropagate() {
            this->grad = 1.0;

            int index = 0;
            std::unordered_set<Value::impl*> explored;
            std::unordered_map<int, Value::impl*> propagateOrder;

            std::shared_ptr<impl> current = getPtr();
            this->sortIntoTopologicalOrder(current, explored, propagateOrder, index);

            for (int i=index-1; i>-1; i--) {
                propagateOrder[i]->backward();
            }
        }
    };


    Value(const double &value, const std::shared_ptr<impl>& a, std::string op = "", const std::shared_ptr<impl>& b = nullptr) {
        pimpl = std::make_shared<impl>(*this, value, a, op, b);
    }

public:
    std::shared_ptr<impl> pimpl;
    Value() {
        pimpl = std::make_shared<impl>();
    }

    Value(const double &value) {
        pimpl = std::make_shared<impl>(value);
    }
    
    Value (const double &value, const std::string &s) {
        pimpl = std::make_shared<impl>(value, s);
    }

    void setData(const double &d) {
        pimpl->data = d;
    };

    double data() {
        return pimpl->data;
    }

    void setLabel(const std::string &s) {
        pimpl->label = s;
    }

    std::string label() {
        return pimpl->label;
    }
    
    std::string op() {
        return pimpl->op;
    }

    void backpropagate() {
        pimpl->backpropagate();
    }

    template <typename T>
    Value raiseTo(T value) {
        return pimpl->raiseTo(value);
    }

    friend Value operator+(const double &value, Value &other) {
        Value v = Value(value);
        return v + other;
    }

    template <typename T>
    Value operator+(T value) {
        return *(this->pimpl) + value;
    }
    
    friend Value operator-(const double &value, Value &other) {
        Value v = Value(value);
        return v - other;
    }

    template <typename T>
    Value operator-(T value) {
        return *(this->pimpl) - value;
    }
    
    friend Value operator*(const double &value, Value &other) {
        Value v = Value(value);
        return v * other;
    }

    template <typename T>
    Value operator*(T value) {
        return *(this->pimpl) * value;
    }
    
    friend Value operator/(const double &value, Value &other) {
        Value v = Value(value);
        return v / other;
    }

    template <typename T>
    Value operator/(T value) {
        return *(this->pimpl) / value;
    }
    
    Value exp() {
        return pimpl->exp();
    }

    Value tanh() {
        return pimpl->tanh();
    }

    void generateGraph(std::string filePath) {
        ValueNetwork n(*this);
        n.createGraph(filePath.data());
    }
};

}