#pragma once

#include <cmath>
#include <dmc/policy.hpp>
#include <dmc/utils.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
#include <open_spiel/spiel_utils.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace dmc {

enum class UpdateMethod { HEDGE, MULTIPLICATIVE_WEIGHTS };

struct SolverSpec {
  static inline constexpr uint64_t DEFAULT_MAX_STEPS = 100000;
  static inline constexpr double DEFAULT_PLAYER_TRAVERSALS = 1;
  static inline constexpr double DEFAULT_BASELINE_TRAVERSALS = 1;
  static inline constexpr uint64_t DEFAULT_BASELINE_START_STEP = 0;
  static inline constexpr uint64_t DEFAULT_PLAYER_UPDATE_FREQ = 1;

  static inline constexpr double DEFAULT_LR_INIT = 1e-3;
  static inline constexpr double DEFAULT_LR_END = 1e-6;
  static inline constexpr uint64_t DEFAULT_DECAYS_STEPS = 1000;
  static inline constexpr double DEFAULT_DECAY_RATE = 1.0;
  static inline constexpr double DEFAULT_GRADIENT_CLIPPING_VALUE = 0.0;
  static inline constexpr double DEFAULT_LOGITS_THRESHOLD = 2.0;
  static inline constexpr double DEFAULT_WEIGHT_DECAY = 0.01;
  static inline constexpr double DEFAULT_ENTROPY_COST = 0.0;

  static inline constexpr UpdateMethod DEFAULT_UPDATE = UpdateMethod::HEDGE;

  static inline constexpr double DEFAULT_EPSILON = 0.1;
  static inline constexpr double DEFAULT_ETA = 1.0;

  uint64_t max_steps = DEFAULT_MAX_STEPS;
  uint32_t player_traversals = DEFAULT_PLAYER_TRAVERSALS;
  uint32_t baseline_traversals = DEFAULT_BASELINE_TRAVERSALS;
  uint64_t baseline_start_step = DEFAULT_BASELINE_START_STEP;
  uint64_t player_update_freq = DEFAULT_PLAYER_UPDATE_FREQ;

  double player_lr_init = DEFAULT_LR_INIT;
  double baseline_lr_init = DEFAULT_LR_INIT;
  double player_lr_end = DEFAULT_LR_END;
  double baseline_lr_end = DEFAULT_LR_END;
  uint64_t decay_steps = DEFAULT_DECAYS_STEPS;
  double decay_rate = DEFAULT_DECAY_RATE;
  double gradient_clipping_value = DEFAULT_GRADIENT_CLIPPING_VALUE;
  double logits_threshold = DEFAULT_LOGITS_THRESHOLD;
  double weight_decay = DEFAULT_WEIGHT_DECAY;
  double entropy_cost = DEFAULT_ENTROPY_COST;

  UpdateMethod update_method = DEFAULT_UPDATE;
  double eta = DEFAULT_ETA;
  bool normalize_returns = false;

  double epsilon = DEFAULT_EPSILON;
  int seed = 0;

  double lr(uint64_t step, double init_lr, double end_lr) const {
    return fmax(init_lr * pow(decay_rate, ((double)step) / (double)decay_steps),
                end_lr);
  }

  double player_lr(uint64_t step) const {
    return lr(step, player_lr_init, player_lr_end);
  }

  double baseline_lr(uint64_t step) const {
    return lr(step, baseline_lr_init, baseline_lr_end);
  }

  nlohmann::json to_json() const;
  static SolverSpec from_json(const nlohmann::json &spec_json);
};

struct SolverState {
  AvgTabularPolicy avg_policy;
  uint64_t step = 0;
  std::vector<torch::optim::SGD> player_opts;
  std::vector<torch::optim::SGD> baseline_opts;
};

template <typename Net> class DeepMwCfrSolver {

  struct ValuesBuffer {
    std::vector<torch::Tensor> features, values;
    void clear() { features.clear(); values.clear(); }
  };
  struct UtilityBuffer {
    std::vector<torch::Tensor> features;
    std::vector<open_spiel::Action> actions;
    std::vector<double> utilities;
    void clear() {
      features.clear();
      actions.clear();
      utilities.clear();
    }
  };

  enum class TraversalType { PLAYER, BASELINE };

public:
  using NetPtr = std::shared_ptr<Net>;

  static_assert(std::is_base_of<torch::nn::Module, Net>(),
                "Net has to be torch::nn::Module");

  DeepMwCfrSolver(std::shared_ptr<const open_spiel::Game> game, SolverSpec spec,
                  std::vector<NetPtr> player_nets, torch::Device net_device,
                  std::vector<NetPtr> baseline_nets = {},
                  bool update_tabular_policy = true)
      : game_(game), spec_(std::move(spec)),
        has_baseline_(baseline_nets.size() > 0),
        player_nets_(std::move(player_nets)),
        baseline_nets_(std::move(baseline_nets)), net_device_(net_device),
        rng_(spec_.seed), update_tabular_policy_(update_tabular_policy) {

    if (player_nets_.size() != (size_t)game_->NumPlayers())
      throw std::invalid_argument(
          "There should be the a network for every player");

    if (has_baseline_ && baseline_nets_.size() != player_nets_.size())
      throw std::invalid_argument(
          "Baseline networks should be of the same size as player networks");

    if (!game_->GetType().provides_information_state_tensor)
      throw std::invalid_argument(
          fmt::format("Game: {} does not provide information state tensor",
                      game_->GetType().long_name));

    if (game_->GetType().chance_mode ==
        open_spiel::GameType::ChanceMode::kSampledStochastic)
      throw std::invalid_argument("Solver only accepts deterministic or "
                                  "explicit chance modes for games.");

    if (spec_.normalize_returns)
      returns_normalizer_ = game_->MaxUtility();
    else
      returns_normalizer_ = 1.0;
  }

  SolverState init() const {

    SolverState state{AvgTabularPolicy(*game_)};

    auto player_opt_config = torch::optim::SGDOptions(spec_.player_lr(0))
                                 .weight_decay(spec_.weight_decay);
    for (const NetPtr &player_net : player_nets_)
      state.player_opts.emplace_back(player_net->parameters(),
                                     player_opt_config);

    if (has_baseline_) {
      auto baseline_opt_config = torch::optim::SGDOptions(spec_.baseline_lr(0))
                                     .weight_decay(spec_.weight_decay);
      for (const NetPtr &baseline_net : baseline_nets_)
        state.baseline_opts.emplace_back(baseline_net->parameters(),
                                         baseline_opt_config);
    }

    return state;
  }

  void run_iteration(SolverState &state) {
    // set networks to eval mode
    for (NetPtr &player_ptr : player_nets_)
      player_ptr->eval();
    if (has_baseline_)
      for (NetPtr &baseline_ptr : baseline_nets_)
        baseline_ptr->eval();

    ValuesBuffer strategy_memory_buffer;
    UtilityBuffer utility_memory_buffer;

    for (open_spiel::Player player{0}; player < game_->NumPlayers(); player++) {

      // collect player traversals
      for (uint32_t traversal = 0; traversal < spec_.player_traversals;
           traversal++) {
        // no gradients required during traversals
        torch::NoGradGuard no_grad;
        auto init_state = game_->NewInitialState();
        traverse(TraversalType::PLAYER, player, *init_state, 1.0, 1.0, 1.0,
                 strategy_memory_buffer, utility_memory_buffer, state);
      }
      // update strategy network
      update_player(player, strategy_memory_buffer, state);

      if (has_baseline_) {

        // update baseline 'player_update_freq' times for each player's update
        for (uint32_t upd = 0; upd < spec_.player_update_freq; upd++) {
          // collect baseline traversals
          for (uint32_t traversal = 0; traversal < spec_.baseline_traversals;
               traversal++) {
            // no gradients required during traversals
            torch::NoGradGuard no_grad;
            auto init_state = game_->NewInitialState();
            traverse(TraversalType::BASELINE, player, *init_state, 1.0, 1.0,
                     1.0, strategy_memory_buffer, utility_memory_buffer, state);
          }

          update_baseline(player, utility_memory_buffer, state);
        }
      }
      // clear buffers for the next player
      strategy_memory_buffer.clear();
      utility_memory_buffer.clear();
    }

    state.step++;
  }

private:
  double traverse(TraversalType traversal_type, open_spiel::Player player,
                  open_spiel::State &state, double player_reach_prob,
                  double others_reach_prob, double sample_reach_prob,
                  ValuesBuffer &strategy_memory_buffer,
                  UtilityBuffer &utility_memory_buffer,
                  SolverState &solver_state) {
    using namespace torch::indexing;

    if (state.IsTerminal()) {
      return state.PlayerReturn(player) / returns_normalizer_;
    } else if (state.IsChanceNode()) {
      auto chance_outcomes = state.ChanceOutcomes();
      const auto [chosen_action, sample_action_prob] =
          open_spiel::SampleAction(chance_outcomes, rng_);
      state.ApplyAction(chosen_action);
      return traverse(traversal_type, player, state, player_reach_prob,
                      others_reach_prob * sample_action_prob,
                      sample_reach_prob * sample_action_prob,
                      strategy_memory_buffer, utility_memory_buffer,
                      solver_state);
    }
    SPIEL_CHECK_PROB(sample_reach_prob);

    const open_spiel::Player current_player = state.CurrentPlayer();
    // get information state representation for the current player
    const auto player_features =
        torch::tensor(state.InformationStateTensor(current_player))
            .to(net_device_);

    const auto legal_actions = state.LegalActions(current_player);
    const auto legal_actions_mask =
        torch::tensor(state.LegalActionsMask(current_player), torch::kDouble);

    // calculate strategy and sampling probabilities
    Net &player_net = *player_nets_[current_player];
    auto logits = legal_actions_mask *
                  player_net.forward(player_features.index({None, Slice{}}))
                      .reshape({-1})
                      .toType(torch::kDouble)
                      .to(torch::kCPU);

    auto strategy_probs_tensor =
        utils::get_probabilities(legal_actions, logits);
    const double epsilon = spec_.epsilon;
    const double act_count = legal_actions.size();
    auto sampling_probs_tensor = strategy_probs_tensor;
    if (current_player == player) {
      sampling_probs_tensor =
          epsilon / act_count + (1.0 - epsilon) * strategy_probs_tensor;
    }
    auto strategy_probs = utils::to_vector<double>(strategy_probs_tensor);
    auto sampling_probs = utils::to_vector<double>(sampling_probs_tensor);

    const auto [chosen_action, sample_action_prob] = open_spiel::SampleAction(
        utils::to_actions_and_probs(legal_actions, sampling_probs), rng_);
    const uint32_t action_ind =
        ranges::distance(ranges::begin(legal_actions),
                         ranges::find(legal_actions, chosen_action));
    const double strategy_action_prob = strategy_probs[action_ind];

    // update average policy
    if (current_player == player && traversal_type == TraversalType::PLAYER) {
      solver_state.avg_policy.UpdateStatePolicy(
          state.InformationStateString(player), strategy_probs,
          /*update_coeff=*/player_reach_prob / sample_reach_prob);
    }

    // NOTE: apply action to the state - it's modified so it should not be used
    // again in this function
    state.ApplyAction(chosen_action);
    // recursive call
    const double subgame_utility = traverse(
        traversal_type, player, state,
        player == current_player ? player_reach_prob * strategy_action_prob
                                 : player_reach_prob,
        player != current_player ? others_reach_prob * strategy_action_prob
                                 : others_reach_prob,
        sample_reach_prob * sample_action_prob, strategy_memory_buffer,
        utility_memory_buffer, solver_state);

    // update utility memory buffer
    if (current_player == player && has_baseline_ &&
        traversal_type == TraversalType::BASELINE) {
      utility_memory_buffer.features.push_back(player_features);
      utility_memory_buffer.actions.push_back(chosen_action);
      utility_memory_buffer.utilities.push_back(subgame_utility);
    }

    // caculate estimated baselines
    const torch::Tensor exp_utilities = [&]() {
      if (has_baseline_ && current_player == player &&
          spec_.baseline_start_step <= solver_state.step) {
        Net &baseline_net = *baseline_nets_[player];
        return legal_actions_mask *
               baseline_net.forward(player_features.index({None, Slice{}}))
                   .reshape({-1})
                   .toType(torch::kDouble)
                   .to(torch::kCPU);
      } else {
        return torch::zeros({game_->NumDistinctActions()}, torch::kDouble);
      }
    }();

    // calculate baseline-enhanced sampled utilities
    exp_utilities.index({chosen_action}) +=
        (subgame_utility - exp_utilities.index({chosen_action})) /
        sample_action_prob;
    // calculate this state's utility as product of strategy and exp_utilities
    // vectors
    const double state_utility =
        torch::tensor(strategy_probs, torch::kDouble)
            .dot(exp_utilities.gather(0, torch::tensor(legal_actions)))
            .template item<double>();

    // calculate baseline corrected sampled value for the trained player
    if (current_player == player && traversal_type == TraversalType::PLAYER) {
      torch::Tensor sampled_value =
          others_reach_prob * exp_utilities / sample_reach_prob;
      // update strategy memory buffer
      strategy_memory_buffer.features.push_back(player_features);
      strategy_memory_buffer.values.push_back(sampled_value);
    }

    return state_utility;
  }

  void update_player(open_spiel::Player player, ValuesBuffer &strategy_dataset,
                     SolverState &state) {

    auto &optimizer = state.player_opts[player];
    // set learning rate
    torch::optim::SGDOptions &opt_config =
        dynamic_cast<torch::optim::SGDOptions &>(optimizer.defaults());
    opt_config.lr(spec_.player_lr(state.step));

    Net &player_net = *player_nets_[player];
    player_net.train();

    torch::Tensor zero = torch::zeros(
        {}, torch::TensorOptions().dtype(torch::kFloat32).device(net_device_));

    optimizer.zero_grad();
    auto features = torch::stack(strategy_dataset.features);
    torch::Tensor values =
        spec_.eta *
        torch::stack(strategy_dataset.values).toType(torch::kFloat32).to(net_device_);

    torch::Tensor logits = player_net.forward(features);
    // center logits around mean
    logits = logits - logits.mean(-1, /*keepdim=*/true);

    if (spec_.update_method == UpdateMethod::MULTIPLICATIVE_WEIGHTS) {
      values = torch::log(1.0 + values);
    }

    if (spec_.logits_threshold > 0.0) {
      // logits will be increased for these actions
      torch::Tensor pos_values = torch::max(values, zero);
      // very large logits cannot be increased
      torch::Tensor can_increase =
          logits.le(spec_.logits_threshold).to(logits.dtype());
      // logits will be decreased for these actions
      torch::Tensor neg_values = torch::min(values, zero);
      // very small logits cannot be decreased
      torch::Tensor can_decrease =
          logits.ge(-spec_.logits_threshold).to(logits.dtype());
      values = neg_values * can_decrease + pos_values * can_increase;
    }
    torch::Tensor policy_loss = -(values.detach() * logits).mean();
    torch::Tensor entropy =
        -(torch::softmax(logits, 1) * torch::log_softmax(logits, 1)).mean();

    torch::Tensor loss = policy_loss - spec_.entropy_cost * entropy;
    loss.backward();

    if (spec_.gradient_clipping_value > 0) {
      torch::nn::utils::clip_grad_value_(player_net.parameters(),
                                         spec_.gradient_clipping_value);
    }
    optimizer.step();

    player_net.eval();
  }

  void update_baseline(open_spiel::Player player,
                       UtilityBuffer &utility_dataset, SolverState &state) {

    auto &optimizer = state.baseline_opts[player];
    // set learning rate
    torch::optim::SGDOptions &opt_config =
        dynamic_cast<torch::optim::SGDOptions &>(optimizer.defaults());
    opt_config.lr(spec_.baseline_lr(state.step));

    Net &baseline_net = *baseline_nets_[player];
    baseline_net.train();

    optimizer.zero_grad();

    auto features = torch::stack(utility_dataset.features);
    torch::Tensor batch_inds =
        torch::arange((int64_t)utility_dataset.features.size(), net_device_);
    torch::Tensor action_inds = torch::tensor(
        utility_dataset.actions,
        torch::TensorOptions().dtype(torch::kInt64).device(net_device_));

    torch::Tensor pred_utilities =
        baseline_net.forward(features).index({batch_inds, action_inds});
    torch::Tensor target_utilities = torch::tensor(
        utility_dataset.utilities,
        torch::TensorOptions().dtype(torch::kFloat32).device(net_device_));

    torch::Tensor loss = torch::mse_loss(pred_utilities, target_utilities);
    loss.backward();
    if (spec_.gradient_clipping_value > 0) {
      torch::nn::utils::clip_grad_value_(baseline_net.parameters(),
                                         spec_.gradient_clipping_value);
    }
    optimizer.step();

    baseline_net.eval();
  }

private:
  std::shared_ptr<const open_spiel::Game> game_;
  const SolverSpec spec_;
  const bool has_baseline_;
  double returns_normalizer_;
  std::vector<NetPtr> player_nets_;
  std::vector<NetPtr> baseline_nets_;
  torch::Device net_device_;
  std::mt19937 rng_;
  const bool update_tabular_policy_;
};

} // namespace dmc
