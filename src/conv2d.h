#pragma once
#include "value.h"
#include "value_tensor.h"
#include "mlp.h"
#include <vector>



namespace grad {
    class Conv2dLayer {
    private:
        int inputHeight;
        int inputWidth;
        int inputChannelCount;
        int kernelSize;
        int strideHeight;
        int strideWidth;
        
        ValueTensor input;
        ValueTensor kernels;
        ValueTensor output;

        void im2col(ValueTensor& input, ValueTensor& output) {
            
        }
        
    public:
        Conv2dLayer(int inputChannelCount, int inputHeight, int inputWidth, int outputChannelCount, int kernelSize, int strideHeight = 1, int strideWidth = 1);
        void passInput(ValueTensor& input);
    };
}