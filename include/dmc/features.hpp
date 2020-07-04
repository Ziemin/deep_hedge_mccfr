#pragma once
#include <vector>

#include <torch/torch.h>

namespace dmc::features {

struct RawInfoStateBuilder {

  torch::Tensor build(const std::vector<double> &info_state_tensor,
                      const torch::Device &device) const {
    return torch::tensor(info_state_tensor).to(device);
  }

  torch::Tensor batch(std::vector<torch::Tensor> examples) const {
    return torch::stack(examples, 0);
  }

  torch::Tensor batch(torch::Tensor example) const {
    using namespace torch::indexing;
    return example.index({None, Slice{}});
  }

};

} // namespace dmc::features
