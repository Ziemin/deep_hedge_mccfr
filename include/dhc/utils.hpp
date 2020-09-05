#pragma once
#include <absl/random/random.h>
#include <algorithm>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>
#include <vector>

namespace dhc::utils {

template <typename DType>
std::vector<DType> to_vector(const torch::Tensor &tensor) {
  torch::Tensor cpu_tensor = tensor.to(torch::kCPU);
  DType *data_arr = cpu_tensor.data_ptr<DType>();
  return ranges::make_subrange(data_arr, data_arr + tensor.numel()) |
         ranges::to<std::vector>();
}

torch::Tensor
get_probabilities(const std::vector<open_spiel::Action> &legal_actions,
                  const torch::Tensor &logits);

open_spiel::ActionsAndProbs
to_actions_and_probs(const std::vector<open_spiel::Action> &actions,
                     const std::vector<double> &probabilities);

template <typename Net>
open_spiel::ActionsAndProbs eval_player_network(Net &player_net,
                                                const open_spiel::State &state,
                                                torch::Device device) {

  using namespace torch::indexing;
  const open_spiel::Player current_player = state.CurrentPlayer();
  const auto player_features =
      torch::tensor(state.InformationStateTensor(current_player))
          .to(device)
          .index({None, Slice{}});
  const auto legal_actions = state.LegalActions(current_player);
  const auto legal_actions_mask =
      torch::tensor(state.LegalActionsMask(current_player), torch::kDouble);

  const auto logits = legal_actions_mask * player_net.forward(player_features)
                                               .reshape({-1})
                                               .toType(torch::kDouble)
                                               .to(torch::kCPU);

  const auto probs =
      to_vector<double>(utils::get_probabilities(legal_actions, logits));
  return to_actions_and_probs(legal_actions, probs);
}

} // namespace dhc::utils
