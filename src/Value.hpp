
#include <iostream>
#include <vector>
#include <set>
#include <string>

class Value 
{
private:
   float data; 
   // Derivative of final evaluation of expression with respect to this Value)
   float grad = 0; // Assume gradient 0 by default
   std::set<Value*> prev;
   std::string op;
public:
    Value(float data, std::set<Value*> children = {}, std::string op = "") {
        this->data = data;
        this->prev = children;
        this->op = op; 
    }
    
    Value operator+(Value &value) {
        float sum = this->data + value.getData();
        Value v = Value(sum, std::set<Value*>({this, &value}), "+");
        return v;
    }

    auto operator-(Value &value) {
        float difference = this->data - value.getData();
        Value v = Value(difference, std::set<Value*>({this, &value}), "-");
        return v;
    }

    auto operator*(Value &value) {
        float product = this->data * value.getData();
        Value v = Value(product, std::set<Value*>({this, &value}), "*");
        return v;
    }

    friend std::ostream & operator << (std::ostream &out, Value &v) {
        out << "Value(data=" << v.getData() << ")" << std::endl;
        return out;
    }

    float getData() { return this->data; }

    void setGrad(float g) { this->grad = g; }
};