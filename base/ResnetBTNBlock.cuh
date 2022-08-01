//
// Created by Dylan on 7/31/2022.
//

#ifndef YOLOPROTOTYPE_RESNETBTNBLOCK_CUH
#define YOLOPROTOTYPE_RESNETBTNBLOCK_CUH

#include <torch/torch.h>

using namespace torch;
namespace base {
    /**
     * ResnetBTNBlock is a block that is used in the Resnet50 and Resnet101 networks.
     * It has 3 conv layers with the "bottleneck" setup
     */
    class ResnetBTNBlock : public nn::Module {
    public:
        std::shared_ptr<nn::Conv2dImpl> conv1;
        std::shared_ptr<nn::Conv2dImpl> conv2;
        std::shared_ptr<nn::Conv2dImpl> conv3;
        
        std::shared_ptr<nn::BatchNorm2dImpl> bn1;
        std::shared_ptr<nn::BatchNorm2dImpl> bn2;
        std::shared_ptr<nn::BatchNorm2dImpl> bn3;
        
        std::shared_ptr<nn::Conv2dImpl> convShortCut;
        bool isConvShortcut;
        
        ResnetBTNBlock(int inChannels, int bottleneckChannels, int outChannels, bool isConvShortcut = false){
            assert((!isConvShortcut) && (inChannels == outChannels));
            this->isConvShortcut = isConvShortcut;
            
            //initialize first bottleneck layer
            conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(
                inChannels, bottleneckChannels, 1)
                .stride(1)
                .padding(0)
            ));
            
            bn1 = register_module("bn1", nn::BatchNorm2d(bottleneckChannels));
            
            //initialize conv layer
            conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(
                bottleneckChannels, bottleneckChannels, 3)
                .stride(1)
                .padding(1)
            ));
            
            bn2 = register_module("bn2", nn::BatchNorm2d(bottleneckChannels));
            
            //initialize second bottleneck layer
            conv3 = register_module("conv3", nn::Conv2d(nn::Conv2dOptions(
                bottleneckChannels, outChannels, 1)
                .stride(1)
                .padding(0)
            ));
            
            bn3 = register_module("bn3", nn::BatchNorm2d(outChannels));
            
            //the conv shortcut (when indicated)
            if(isConvShortcut) {
                convShortCut = register_module("convShortCut", nn::Conv2d(nn::Conv2dOptions(
                        inChannels, outChannels, 1)
                        .stride(1)
                        .padding(0)
                ));
                
                bn3 = register_module("bn3", nn::BatchNorm2d(outChannels));
            }
        }
    
        Tensor forward(Tensor x) const;
    };
}


#endif //YOLOPROTOTYPE_RESNETBTNBLOCK_CUH
