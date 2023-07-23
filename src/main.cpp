#include <iostream>
#include "value.hpp"
#include "neuron.hpp"

using namespace grad;

int main() {
    
    Value x1 = Value(2.0, "x1");
    Value x2 = Value(0.0, "x2");

    Value w1 = Value(-3.0, "w1");
    Value w2 = Value(1.0, "w2");
    
    Value b = Value(6.8813735870195432, "b");

    Value x1w1 = x1*w1; x1w1.setLabel("x1w1");
    Value x2w2 = x2*w2; x2w2.setLabel("x2w2");
    Value x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.setLabel("x1w1x2w2");
    
    Value n = x1w1x2w2 + b; n.setLabel("n");
    
    Value o = n.tanh();
    
    o.backpropagate();

    ValueNetwork network(o);

    char filePath[] = "./test.png";
    // network.createGraph(filePath);

    MLP mlp(3,{4,4,1});
    std::vector<Value> inputs = {2.0, 3.0, 4.0};
    std::cout << inputs[0].data() << " " << inputs[1].data() << " " << inputs[2].data() << std::endl;
    std::vector<Value> outputs = mlp.passInputs(inputs);
    ValueNetwork mlpNet(outputs[0]);
    // mlpNet.createGraph(filePath);
    
    std::cout << outputs[0].data() << std::endl;
    std::cout << "IT DIDN'T BREAK!!!" << std::endl;

    return 0;
}