set (CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION "3.19.2")

project(mgrad)

link_directories(${CMAKE_SOURCE_DIR}/lib)
add_executable(mgrad src/utils.cpp src/value.cpp src/value_network.cpp src/neuron.cpp src/layer.cpp src/mlp.cpp src/main.cpp)
# add_executable(mgrad src/main.cpp)

target_include_directories(mgrad PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mgrad cdt cgraph gvc)