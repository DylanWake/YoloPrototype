#include <iostream>
#include <torch/torch.h>

using namespace torch;

int main() {
    Tensor test = torch::randn({2, 3}).to(torch::device(torch::Device("cuda")));
    float testDa[] = {1,2,3,4,5,6};
    cudaMemcpy(test.data_ptr(), testDa, sizeof(float) * 6, cudaMemcpyHostToDevice);
    std::cout << test << std::endl;
    return 0;
}
