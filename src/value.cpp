#include "value.h"
#include <math.h>
#include <iostream>

namespace grad {
    void Value::impl::backward() {
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
        } else if (this->op == "relu") {
            if (this->data == 0) {
                prev[0]->grad += 0.0 * this->grad;
            } else {
                prev[0]->grad += 1.0 * this->grad;
            }
        } else if (this->op == "^") {
            double exponent = prev[1]->data;
            prev[0]->grad += exponent * (std::pow(prev[0]->data, exponent-1)) * this->grad; 
        }
    }

    void Value::impl::sortIntoTopologicalOrder(std::shared_ptr<impl> &current, 
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

    Value::impl::impl(){
        this->data = 0.0;
        this->grad = 0.0;
    }
    
    Value::impl::impl(double data) {
        this->data = data;
        this->grad = 0.0;
    }

    Value::impl::impl(double data, std::string label) : impl(data) {
        // this->data = data;
        this->label = label;
    }

    Value::impl::impl(
        Value &outer, 
        double data, 
        const std::shared_ptr<impl>& a, 
        std::string op, 
        const std::shared_ptr<impl>& b /*=nullptr*/
        ) {

        this->data = data;
        prev[0] = std::shared_ptr<impl>(a);
        if (b != nullptr) {
            prev[1] = std::shared_ptr<impl>(b);
        }
        this->op = op;
    }

    std::shared_ptr<Value::impl> Value::impl::getPtr() {
        return shared_from_this();
    }

    std::shared_ptr<Value::impl>& Value::impl::getChild(const int &i) {
        return this->prev[i];
    }
    
    Value Value::impl::raiseTo(const double &value) {
        Value v = Value(value);
        return this->raiseTo(v);
    }
    
    Value Value::impl::raiseTo(Value &value) {
        double power = std::pow(this->data, value.data());
        Value v = Value(power, this->getPtr(), "^", value.pimpl);
        return v;
    }

    Value Value::impl::operator+(const double &value) {
        Value v = Value(value);
        return (*this) + v;
    }

    Value Value::impl::operator+(Value &&value) {
        double sum = this->data + value.data();
        Value v = Value(sum, this->getPtr(), "+", value.pimpl);
        return v;
    }

    Value Value::impl::operator+(Value &value) {
        double sum = this->data + value.data();
        Value v = Value(sum, this->getPtr(), "+", value.pimpl);
        return v;
    }

    Value Value::impl::operator-(const double &value) {
        Value v = Value(value);
        return (*this) - v;
    }

    Value Value::impl::operator-(Value &&value) {
        double sum = this->data - value.data();
        Value v = Value(sum, this->getPtr(), "-", value.pimpl);
        return v;
    }

    Value Value::impl::operator-(Value &value) {
        double sum = this->data - value.data();
        Value v = Value(sum, this->getPtr(), "-", value.pimpl);
        return v;
    }

    Value Value::impl::operator*(const double &value) {
        Value v = Value(value);
        return (*this) * v;
    }

    Value Value::impl::operator*(Value &&value) {
        double sum = this->data * value.data();
        Value v = Value(sum, this->getPtr(), "*", value.pimpl);
        return v;
    }

    Value Value::impl::operator*(Value &value) {
        double sum = this->data * value.data();
        Value v = Value(sum, this->getPtr(), "*", value.pimpl);
        return v;
    }

    Value Value::impl::operator/(const double &value) {
        Value v = Value(value);
        return (*this) / v;
    }

    Value Value::impl::operator/(Value &&value) {
        Value v = value.raiseTo(-1.0);
        return (*this) * v;
    }

    Value Value::impl::operator/(Value &value) {
        Value v = value.raiseTo(-1.0);
        return (*this) * v;
    }

    Value Value::impl::exp() {
        double x = this->data;
        // Value current = *this;
        Value v = Value(std::exp(x), this->getPtr(), "exp");
        return v;
    }

    Value Value::impl::tanh() {
        double x = this->data;
        double t = (std::exp(2*x) - 1) / (std::exp(2*x) + 1);

        Value v = Value(t, this->getPtr(), "tanh");
        return v;
    }

    Value Value::impl::relu() {
        Value v = Value(std::max(0.0, this->data), this->getPtr(), "relu");
        return v;
    }

    void Value::impl::backpropagate() {
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
    
    Value::Value(const double &value, const std::shared_ptr<impl>& a, std::string op /*=""*/, const std::shared_ptr<impl>& b /*=nullptr*/) {
        pimpl = std::make_shared<impl>(*this, value, a, op, b);
    }

    Value::Value() {
        pimpl = std::make_shared<impl>();
    }

    Value::Value(const double &value) {
        pimpl = std::make_shared<impl>(value);
    }
    
    Value::Value (const double &value, const std::string &s) {
        pimpl = std::make_shared<impl>(value, s);
    }

    void Value::setData(const double &d) {
        pimpl->data = d;
    };

    void Value::setGrad(const double &g) {
        pimpl->grad = g;
    }

    void Value::increaseGrad(const double &g) {
        pimpl->grad += g;
    }

    double Value::data() {
        return pimpl->data;
    }

    double Value::grad() {
        return pimpl->grad;
    }

    void Value::setLabel(const std::string &s) {
        pimpl->label = s;
    }

    std::string Value::label() {
        return pimpl->label;
    }
    
    std::string Value::op() {
        return pimpl->op;
    }

    void Value::backward() {
        pimpl->backpropagate();
    }

    Value operator+(const double &value, Value &other) {
        Value v = Value(value);
        return v + other;
    }

    Value operator+(const double &value, Value &&other) {
        Value v = Value(value);
        return v + other;
    }
    
    Value operator-(const double &value, Value &other) {
        Value v = Value(value);
        return v - other;
    }

    Value operator-(const double &value, Value &&other) {
        Value v = Value(value);
        return v - other;
    }
    
    Value operator*(const double &value, Value &other) {
        Value v = Value(value);
        return v * other;
    }

    Value operator*(const double &value, Value &&other) {
        Value v = Value(value);
        return v * other;
    }
    
    Value operator/(const double &value, Value &other) {
        Value v = Value(value);
        return v / other;
    }

    Value operator/(const double &value, Value &&other) {
        Value v = Value(value);
        return v / other;
    }
    
    Value Value::exp() {
        return pimpl->exp();
    }

    Value Value::tanh() {
        return pimpl->tanh();
    }

    Value Value::relu() {
        return pimpl->relu();
    }

    std::ostream& operator<<(std::ostream& o, Value& v) {
        o << "Value(data=" << v.data() << ")";
        return o;
    }
}