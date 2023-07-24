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
    friend class Value;
    friend class ValueNetwork;
    
    Utils();
};

std::random_device Utils::rd;
std::mt19937 Utils::generator = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());

}