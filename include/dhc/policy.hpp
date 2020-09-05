#pragma once
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <cmath>
#include <dhc/utils.hpp>
#include <numeric>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
#include <open_spiel/spiel_utils.h>
#include <unordered_map>
#include <vector>

namespace dhc {

class AvgTabularPolicy : public open_spiel::TabularPolicy {
public:
  friend class boost::serialization::access;

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
                         double update_coeff) {
    auto &state_policy = PolicyTable()[info_state];
    if (state_policy.size() != latest_probs.size()) {
      open_spiel::SpielFatalError(
          "Policy to be updated has different size than the latest policy");
    }
    auto &state_bits = c_bits_[info_state];

    for (size_t ind = 0; ind < state_policy.size(); ind++) {
      double increment = update_coeff * latest_probs[ind];

      SPIEL_CHECK_FALSE(std::isnan(increment) || std::isinf(increment));

      // Kahan summation algorithm
      double y = increment - state_bits[ind];
      double t = state_policy[ind].second + y;
      state_bits[ind] = (t - state_policy[ind].second) - y;
      state_policy[ind].second = t;
    }
  }

private:
  template <class Archive>
  void serialize(Archive &ar, const unsigned int /*version*/) {
    auto &policy_table = PolicyTable();
    ar & policy_table;
    ar & c_bits_;
  }

  // lost bits for the Kahan summation algorithm
  std::unordered_map<std::string, std::vector<double>> c_bits_;
};


template <typename Net>
class NeuralPolicy : public open_spiel::Policy {
public:
  NeuralPolicy(std::vector<std::shared_ptr<Net>> player_nets,
               torch::Device net_device)
      : player_nets_(std::move(player_nets)),
        net_device_(net_device) {}

  open_spiel::ActionsAndProbs
  GetStatePolicy(const open_spiel::State &state) const override {
    const open_spiel::Player current_player = state.CurrentPlayer();
    Net &player_net = *player_nets_[current_player];
    return utils::eval_player_network(player_net, state, net_device_);
  }

private:
  mutable std::vector<std::shared_ptr<Net>> player_nets_;
  torch::Device net_device_;
};

} // namespace dhc
