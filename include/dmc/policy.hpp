#pragma once
#include <open_spiel/spiel_utils.h>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
#include <vector>

namespace dmc {

class AvgTabularPolicy : public open_spiel::TabularPolicy {
public:
  AvgTabularPolicy(const open_spiel::Game &game)
      : open_spiel::TabularPolicy(game) {
    for (const auto &[info_state, state_policy] : PolicyTable()) {
      c_bits_[info_state] = std::vector<double>(state_policy.size(), 0.0);
    }
  }
  AvgTabularPolicy(const AvgTabularPolicy &other) = default;

  open_spiel::ActionsAndProbs
  GetStatePolicy(const std::string &info_state) const override {
    auto &policy_table = PolicyTable();
    auto iter = policy_table.find(info_state);
    if (iter == policy_table.end()) {
      return {};
    } else {
      // normalize probabilities
      open_spiel::ActionsAndProbs actions_and_probs = iter->second;
      double prob_sum = 0.0;
      for (auto [act, prob] : actions_and_probs)
        prob_sum += prob;
      for (auto &[act, prob] : actions_and_probs)
        prob /= prob_sum;
      return actions_and_probs;
    }
  }

  void UpdateStatePolicy(const std::string &info_state,
                         const std::vector<double> &latest_probs,
                         double player_reach_prob, double sample_reach_prob) {
    auto &state_policy = PolicyTable()[info_state];
    if (state_policy.size() != latest_probs.size()) {
      open_spiel::SpielFatalError(
          "Policy to be updated has different size than the latest policy");
    }
    auto &state_bits = c_bits_[info_state];

    for (size_t ind = 0; ind < state_policy.size(); ind++) {
      double increment =
          player_reach_prob * latest_probs[ind] / sample_reach_prob;

      SPIEL_CHECK_FALSE(std::isnan(increment) || std::isinf(increment));

      // Kahan summation algorithm
      double y = increment - state_bits[ind];
      double t = state_policy[ind].second + y;
      state_bits[ind] = (t - state_policy[ind].second) - y;
      state_policy[ind].second = t;
    }
  }

private:
  // lost bits for the Kahan summation algorithm
  std::unordered_map<std::string, std::vector<double>> c_bits_;
};
} // namespace dmc
