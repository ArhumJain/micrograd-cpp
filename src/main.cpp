#include <iostream>
#include "value.h"
#include "value_network.h"
#include "mlp.h"
#include "value.h"
#include "value_tensor.h"
#include <vector>

#include <chrono>
// #include "value_network.hpp"
// #include "value.hpp"
// #include "neuron.hpp"

using namespace grad;

using namespace std::chrono;
int main() {
    // Value x1 = Value(2.0, "x1");
    // Value x2 = Value(0.0, "x2");

    // Value w1 = Value(-3.0, "w1");
    // Value w2 = Value(1.0, "w2");
    
    // Value b = Value(6.8813735870195432, "b");

    // Value x1w1 = x1*w1; x1w1.setLabel("x1w1");
    // Value x2w2 = x2*w2; x2w2.setLabel("x2w2");
    // Value x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.setLabel("x1w1x2w2");
    
    // Value n = x1w1x2w2 + b; n.setLabel("n");
    
    // Value o = n.tanh();
    
    // o.backward();

    // ValueNetwork network(o);

    // char filePath[] = "./test.png";
    // network.createGraph(filePath);
    
    Value x = 5.0;
    Value y = 1.0;

    // Value z = 1.0 + x * 10.0 / 69.0 + y + x + y.raiseTo(5.0);

    // std::cout << z << std::endl;
    std::vector<std::vector<Value>> xTrain = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    
    std::vector<Value> targets = {1.0, -1.0, -1.0, 1.0};

    MLP network = MLP(3, {4,4,1});
    
    std::vector<Value> yPred;

    for (int i=0; i<20; i++) {
        yPred.clear();
        for (auto& train: xTrain) {
            yPred.push_back(network.passInputs(train)[0]);
        }

        Value loss = Loss::meanSquaredError(yPred, targets);
        
        std::cout << "Loss: " << loss << std::endl;
        // std::cout << "{";
        // for (auto& pred: yPred) {
        //     std::cout << pred.data() << ",";
        // }
        // std::cout << "}" << std::endl;

        network.zeroGrad();
        std::cout << network.layers[0].neurons[0].weights[0].data() << std::endl;
        loss.backward();
        std::cout << network.layers[0].neurons[0].weights[0].grad() << std::endl;

        network.updateParameters(-0.1);
    }
    
    return 0;
}