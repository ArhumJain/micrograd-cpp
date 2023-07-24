#pragma once
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>

namespace grad {
    class Value {
        friend class ValueNetwork;
    private:
        class impl : public std::enable_shared_from_this<impl> {
        private:
            std::shared_ptr<impl> prev[2] = {nullptr, nullptr};

            void backward();
            void sortIntoTopologicalOrder(std::shared_ptr<impl> &current, 
                                      std::unordered_set<Value::impl*> &explored,
                                      std::unordered_map<int, Value::impl*> &order, 
                                      int &currentIndex);
        public:
            double data;
            double grad = 0.0;
            std::string label = "";
            std::string op = "";

            impl();
            impl(double data);
            impl(double data, std::string label);
            impl(
                Value &outer, 
                double data, 
                const std::shared_ptr<impl>& a, 
                std::string op, 
                const std::shared_ptr<impl>& b = nullptr
                );
            
            std::shared_ptr<impl> getPtr();
            std::shared_ptr<impl>& getChild(const int &i);
            
            Value raiseTo(const double &value);
            Value raiseTo(Value &value);

            Value operator+(const double &value);
            Value operator+(Value &&value);
            Value operator+(Value &value);

            Value operator-(const double &value);
            Value operator-(Value &&value);
            Value operator-(Value &value);

            Value operator*(const double &value);
            Value operator*(Value &&value);
            Value operator*(Value &value);

            Value operator/(const double &value);
            Value operator/(Value &&value);
            Value operator/(Value &value);
            
            Value exp();
            Value tanh();

            void backpropagate();
        };
    
    Value(const double &value, const std::shared_ptr<impl>& a, std::string op = "", const std::shared_ptr<impl>& b = nullptr);
    public:
        std::shared_ptr<impl> pimpl;

        Value();
        Value(const double &value);
        Value(const double &value, const std::string &s);

        void setData(const double &d);
        void setGrad(const double &g);
        void increaseGrad(const double &g);
        double data();
        double grad();
        void setLabel(const std::string &s);
        std::string label();
        std::string op();
        void backward();

        template <typename T>
        Value raiseTo(T value) {
            return pimpl->raiseTo(value);
        }
        
        friend Value operator+(const double &value, Value &other);
        friend Value operator+(const double &value, Value &&other);

        template <typename T>
        Value operator+(T value) {
            return *(this->pimpl) + value;
        }

        friend Value operator-(const double &value, Value &other);
        friend Value operator-(const double &value, Value &&other);

        template <typename T>
        Value operator-(T value) {
            return *(this->pimpl) - value;
        }
        
        friend Value operator*(const double &value, Value &other);
        friend Value operator*(const double &value, Value &&other);

        template <typename T>
        Value operator*(T value) {
            return *(this->pimpl) * value;
        }
        
        friend Value operator/(const double &value, Value &other);
        friend Value operator/(const double &value, Value &&other);
        
        template <typename T>
        Value operator/(T value) {
            return *(this->pimpl) / value;
        }
        
        Value exp();
        Value tanh();
        
        friend std::ostream& operator<<(std::ostream& o, Value& v);
        
    
    };
}