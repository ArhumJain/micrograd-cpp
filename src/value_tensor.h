#pragma once
#include "value.h"
#include <vector>
#include <tuple>
namespace grad {
    class ValueTensor {
    private:
        int axesCount;
    public:
        std::vector<int> axesSizes;
        std::vector<int> strides;
        std::vector<Value> values;
        
        template<typename... Args>
        ValueTensor(Args&&... sizes) {
            ((axesSizes.push_back(sizes)), ...);
            // std::cout << "Dimension Sizes: ";
            // for (auto& i: dimensionSizes) {
            //     std::cout << i << " ";
            // }
            // std::cout << std::endl;
            this->values = std::vector<Value>((sizes * ...));
            this->strides = std::vector<int>(this->axesSizes.size(), 1);
            // std::cout << "Strides size: " << strides.size() << std::endl;
            for (int i=this->axesSizes.size()-2; i>-1; i--) {
                this->strides[i] = this->strides[i+1] * this->axesSizes[i];
            }
            // std::cout << strides[0] << std::endl;
        }

        size_t sizeOfAxis(int axis) const {
            if (axis < 0 || axis >= this->axesCount) {
                throw std::length_error("Attemp to access out-of-bounds axis");
            }
            return axesSizes[axis];
        }

        template<typename... Args> 
        Value& operator()(Args&&... indices) const {
            int i = 0;
            int offset = 0;
            // std::cout << "POINT TEST" << std::endl;
            // std::cout << strides[0] << std::endl;
            auto setOffset = [&,this](int index) {
                offset += this->strides[i] * index;
                i++;
            };
            (setOffset(std::forward<Args>(indices)), ...);
            return this->values[offset];
        }
    };
}