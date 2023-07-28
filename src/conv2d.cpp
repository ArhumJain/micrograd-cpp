#pragma once
#include "conv2d.h"

namespace grad {
    Conv2dLayer::Conv2dLayer(int inputChannelCount, int inputHeight, int inputWidth, int outputChannelCount, int kernelSize, int strideHeight, int strideWidth) {
        this->inputChannelCount = inputChannelCount;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->strideHeight = strideHeight;
        this->strideWidth = strideWidth;
        
        this->input = ValueTensor(inputChannelCount, inputHeight, inputWidth);

        int outputHeight = static_cast<int>((inputHeight - kernelSize)/strideHeight) + 1;
        int outputWidth = static_cast<int>((inputWidth - kernelSize)/strideWidth) + 1;
        this->output = ValueTensor(outputChannelCount, outputHeight, outputWidth);

        this->kernels = ValueTensor(outputChannelCount, inputChannelCount, kernelSize, kernelSize);
    }

    void Conv2dLayer::im2col(ValueTensor& input, ValueTensor& output) {
        output = ValueTensor(this->kernelSize * this->kernelSize * this->inputChannelCount, this->inputHeight * this->inputWidth);
        for (int i=0; i<this->inputHeight; i+=this->strideHeight) {
            for (int j=0; j<this->inputWidth; j+=this->strideWidth) {
                
            }
        }
    }
    void Conv2dLayer::passInput(ValueTensor &input) {

        ValueTensor im2colTransform;
        this->im2col(input, im2colTransform);

        
        
    };
}