#pragma once
#include "value.h"
#include "mlp.h"
#include <vector>



namespace grad {
    template <class T>
    class ValueBuffer {
    private:
        int dimensionCount;
        int* dimensionSizes;
        Value* values;

        void findOffset(int& offset, int& dimension, int index) {
            offset += this->dimensionSizes[dimension] * index;
        }
        
    public:
        template<typename sizes...>
        ValueBuffer(std::initializer_list<int> sizes) {
            for ()
        }

        template<typename Args...>
        Value& operator()(Args&&... indices) {
            int i = 0;
            int offset = 0;
            updateOffset(offset, i++, std::forward<Args>(indices)), ...;
            return values[offset];
        }
    };
    class Conv2dLayer {
        int inputHeight;
        int inputWidth;
        int inputChannelCount;
        int kernelSize;
        Value* input;
        Value* kernels;
        Value* output;
        
        Conv2dLayer(int inputChannelCount, int inputHeight, int inputWidth, int outputChannelCount, int kernelSize, Value* input);
    };
}