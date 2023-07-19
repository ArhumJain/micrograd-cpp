#pragma once

#include <string>
#include <random>
#include <fstream>
#include <cstring>
#include <ctime>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
#include "value.hpp"

class Value;

class ValueNetwork {
private:
    Value* node;
    
    std::string randString() {
        const std::string CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        static std::random_device random_device;
        static std::mt19937 generator(random_device());

        std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

        std::string random_string;

        for (std::size_t i = 0; i < 10; ++i)
        {
            random_string += CHARACTERS[distribution(generator)];
        }

        return random_string;
    }
    
    void buildGraph(Agraph_t *G, Value* currentNode, Agnode_t *parent = nullptr) {
        char attributeLabel[] = "label";
        char attributeShape[] = "shape";
        char shapeEllipse[] = "ellipse";
        char labelEdge[] = "edge";
        
        Agnode_t *current = agnode(G, &randString()[0], 1);
        std::string d = std::to_string(currentNode->data);
        std::string grad = std::to_string(currentNode->grad);
        std::string output = (currentNode->label == "" ? "" : currentNode->label + "\n") + "Data: " + d + "\nGrad: " + grad;
        agset(current, attributeLabel, &output[0]);
        
        if (parent != nullptr) {
            Agedge_t *currentToParent = agedge(G, current, parent, labelEdge, 1);
        }

        if (currentNode->op == "") {
            return;
        }
        
        Agnode_t* operation = agnode(G, &randString()[0], 1);
        agset(operation, attributeShape, shapeEllipse);
        agset(operation, attributeLabel, &currentNode->op[0]);
        Agedge_t* opToCurrent = agedge(G, operation, current, labelEdge, 1);

        for (int i=0; i<2; i++) {
            Value* child = currentNode->prev[i];
            if (child != nullptr) {
                this->buildGraph(G, child, operation);
            }
        }
    }
    
public:
    ValueNetwork(Value *head) {
        this->node = head;
    }

    void createGraph(char filePath[]) {
        Agraph_t *G;
        GVC_t *gvc;

        gvc = gvContext();

        char network[] = "network";
        G = agopen(network, Agdirected, 0);

        char rankdir[] = "rankdir"; char LR[] = "LR";
        agattr(G, AGRAPH, rankdir, LR);
        char dpi[] = "dpi"; char dpiValue[] = "512";
        agattr(G, AGRAPH, dpi, dpiValue);
        char shape[] = "shape"; char circle[] = "circle";
        agattr(G, AGNODE, shape, circle);
        char label[] = "label"; char def[] = "default";
        agattr(G, AGNODE, label, def);
        char fixedsize[] = "fixedsize"; char tru[] = "true";
        agattr(G, AGNODE, fixedsize, tru);
        char height[] = "height"; char heightValue[] = "1.3";
        agattr(G, AGNODE, height, heightValue);
        char fontsize[] = "fontsize"; char fontSizeValue[] = "12";
        agattr(G, AGNODE, fontsize, fontSizeValue);
        
        this->buildGraph(G, this->node);

        gvLayout(gvc, G, "dot");
        gvRenderFilename(gvc, G, "png", filePath);
        gvFreeLayout(gvc, G);
        agclose(G);
    }
};