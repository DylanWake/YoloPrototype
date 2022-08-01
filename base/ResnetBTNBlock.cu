//
// Created by Dylan on 7/31/2022.
//

#include "ResnetBTNBlock.cuh"

Tensor base::ResnetBTNBlock::forward(Tensor x) const {
    //first bottleneck layer
    Tensor sc = x.copy_(x);
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = nn::functional::relu(x);
    
    //conv layer
    x = conv2->forward(x);
    x = bn2->forward(x);
    x = nn::functional::relu(x);
    
    //second bottleneck layer
    x = conv3->forward(x);
    x = bn3->forward(x);
    x = nn::functional::relu(x);
    
    //the conv shortcut (when indicated)
    if(isConvShortcut) {
        sc = convShortCut->forward(sc);
        sc = bn3->forward(sc);
        sc = nn::functional::relu(sc);
    }
    
    x += sc;
    
    return x;
}
