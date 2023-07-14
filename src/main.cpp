#include <iostream>
#include "Value.hpp"


int main() {
    Value a = Value(2.0f);
    Value b = Value(-3.0f);
    Value c = Value(10.0);

    Value d = a*b + c;
    
    std::cout << d << std::endl;
    return 0;
}