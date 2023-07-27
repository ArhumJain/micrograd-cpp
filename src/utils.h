#pragma once
#include <random>
#include <chrono>

namespace grad 
{

class Utils {
private:
    static std::random_device rd;
    static std::mt19937 generator;

    friend class Neuron;
    friend class Feature;
    friend class Value;
    friend class ValueNetwork;
    
    Utils();
};

}