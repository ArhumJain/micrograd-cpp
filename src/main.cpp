#include <iostream>
#include <value_network.hpp>
#include "value.hpp"


int main() {
    // Value x1 = Value(2.0); x1.label = "x1";
    // Value x2 = Value(0.0); x2.label = "x2";

    // Value w1 = Value(-3.0); w1.label = "w1";
    // Value w2 = Value(1.0); w2.label = "w2";
    
    // Value b = Value(6.8813735870195432); b.label = "b";

    // Value x1w1 = x1*w1; x1w1.label = "x1w1";
    // Value x2w2 = x2*w2; x2w2.label = "x2w2";
    // Value x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2";

    // Value n = x1w1x2w2 + b; n.label = "n";
    // Value o = n.tanh(); o.label = "o";
    

    Value x = Value(1.0); x.label = "x";
    Value y = Value(9.0); y.label = "y";
    Value a = x + y; a.label = "a";
    Value b = Value(10.0); b.label = "b";
    Value o = a + b; o.label = "o";
    o = 78.0 + o;

    b = Value(99.0);
    
    
    ValueNetwork network(&o);

    char filepath[] = "./test.png";
    network.createGraph(filepath);
    return 0;
}