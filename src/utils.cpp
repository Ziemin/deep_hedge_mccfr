#include <dmc/utils.hpp>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>
#include <vector>

using namespace ranges;

namespace dmc::utils {

torch::Tensor get_probabilities(const std::vector<open_spiel::Action> &legal_actions,
                                const torch::Tensor &logits) {
  if (logits.dim() != 1) {
    throw std::invalid_argument("Tensor should be one-dimensional");
  }
  return logits.gather(0, torch::tensor(legal_actions)).softmax(0);
}

open_spiel::ActionsAndProbs to_actions_and_probs(
     const std::vector<open_spiel::Action> &actions,
     const std::vector<double> &probabilities)
{
  return views::zip(actions, probabilities)
    | views::transform([](auto act_prob_tuple) {
        return std::make_pair(std::get<0>(act_prob_tuple),
                              std::get<1>(act_prob_tuple));
      })
    | ::to<std::vector>();
}

} // namespace dmc::utils
