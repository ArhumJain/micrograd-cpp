#include "utils.h"
#include <random>

namespace grad 
{

Utils::Utils() {}

std::random_device Utils::rd;
std::mt19937 Utils::generator = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());

}