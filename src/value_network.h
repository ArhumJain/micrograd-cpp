#pragma once
#include "value.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <graphviz/gvc.h>
#include <graphviz/cgraph.h>
namespace grad {

    class ValueNetwork {
    private:
        std::shared_ptr<Value::impl> head;
        std::unordered_map<Value::impl*, Agnode_t*> nodes = {};
        
        std::string randString();
        void buildGraph(Agraph_t* G, std::shared_ptr<Value::impl>& currentNode, Agnode_t* parent = nullptr);
    public:
        ValueNetwork(Value &head);

        void createGraph(char filePath[]);
    };
}