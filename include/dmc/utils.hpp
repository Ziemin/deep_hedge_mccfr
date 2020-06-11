#pragma once
#include <absl/random/random.h>
#include <algorithm>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>
#include <vector>

namespace dmc::utils {

template <typename DType>
std::vector<DType> to_vector(const torch::Tensor &tensor) {
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU);
  DType *data_arr = cpu_tensor.data_ptr<DType>();
  return ranges::make_subrange(data_arr, data_arr + tensor.numel()) |
         ranges::to<std::vector>();
}

torch::Tensor get_probabilities(const std::vector<open_spiel::Action> &legal_actions,
                                const torch::Tensor& logits);

open_spiel::ActionsAndProbs to_actions_and_probs(const std::vector<open_spiel::Action> &actions,
                                 const std::vector<double> &probabilities);

} // namespace dmc::utils
