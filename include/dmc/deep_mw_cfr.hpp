#pragma once

#include <dmc/datasets.hpp>
#include <dmc/policy.hpp>
#include <dmc/utils.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <functional>
#include <memory>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
#include <open_spiel/spiel_utils.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace dmc {

using LRSchedule = std::function<double(uint64_t)>;

enum class UpdateMethod { HEDGE, MULTIPLICATIVE_WEIGHTS };

struct SolverSpec {
  static inline constexpr double DEFAULT_EPSILON = 0.1;
  static inline constexpr double DEFAULT_ETA = 0.1;
  static inline constexpr double DEFAULT_PLAYER_TRAVERSALS = 1;
  static inline constexpr double DEFAULT_BASELINE_TRAVERSALS = 1;
  static inline constexpr double DEFAULT_LR_SCHEDULE(uint64_t /*step*/) {
    return 1e-3;
  }
  static inline constexpr UpdateMethod DEFAULT_UPDATE = UpdateMethod::HEDGE;
  static inline constexpr size_t DEFAULT_BATCH_SIZE = 64;
  static inline constexpr double DEFAULT_LOGITS_THRESHOLD = 2.0;
  static inline constexpr double DEFAULT_WEIGHT_DECAY = 0.01;
  static inline constexpr uint64_t DEFAULT_BASELINE_START_STEP = 0;
  static inline constexpr double DEFAULT_ENTROPY_COST = 0.0;
  static inline constexpr uint64_t DEFAULT_PLAYER_UPDATE_FREQ = 1;

  std::shared_ptr<const open_spiel::Game> game;
  torch::Device device;
  LRSchedule player_lr_schedule = DEFAULT_LR_SCHEDULE;
  LRSchedule baseline_lr_schedule = DEFAULT_LR_SCHEDULE;
  double epsilon = DEFAULT_EPSILON;
  double eta = DEFAULT_ETA;
  uint32_t player_traversals = DEFAULT_PLAYER_TRAVERSALS;
  uint32_t baseline_traversals = DEFAULT_BASELINE_TRAVERSALS;
  UpdateMethod update_method = DEFAULT_UPDATE;
  size_t batch_size = DEFAULT_BATCH_SIZE;
  double logits_threshold = DEFAULT_LOGITS_THRESHOLD;
  double weight_decay = DEFAULT_WEIGHT_DECAY;
  double entropy_cost = DEFAULT_ENTROPY_COST;
  uint64_t baseline_start_step = DEFAULT_BASELINE_START_STEP;
  uint64_t player_update_freq = DEFAULT_PLAYER_UPDATE_FREQ;
  bool normalize_returns = false;
  int seed = 0;

};

struct SolverState {
  AvgTabularPolicy avg_policy;
  uint64_t step = 0;
  std::vector<torch::optim::SGD> player_opts;
  std::vector<torch::optim::SGD> baseline_opts;
};


template <typename Net, typename FeaturesBuilder>
class DeepMwCfrSolver {
  using FeaturesType = typename Net::FeaturesType;

  using ValueExample = torch::data::Example<FeaturesType, torch::Tensor>;
  using UtilityExample =
      torch::data::Example<std::tuple<FeaturesType, open_spiel::Action>,
                           double>;

public:
  using NetPtr = std::shared_ptr<Net>;

  static_assert(std::is_base_of<torch::nn::Module, Net>(),
                "Net has to be torch::nn::Module");

  DeepMwCfrSolver(SolverSpec spec, std::vector<NetPtr> player_nets,
                  FeaturesBuilder features_builder,
                  std::vector<NetPtr> baseline_nets = {},
                  bool update_tabular_policy = true)
    : has_baseline_(baseline_nets.size() > 0),
      spec_(std::move(spec)), player_nets_(std::move(player_nets)),
      baseline_nets_(std::move(baseline_nets)),
      features_builder_(std::move(features_builder)), rng_(spec_.seed),
      update_tabular_policy_(update_tabular_policy) {

    static_assert(
        std::is_same<FeaturesType,
                     decltype(features_builder_.build(
                         std::vector<double>{}, torch::Device(torch::kCPU)))>(),
        "FeaturesBuilder should return the same type as is expected "
        "by the networks");

    if (player_nets_.size() != spec_.game->NumPlayers()) {
      throw std::invalid_argument(
          "There should be the a network for every player");
    }
    if (has_baseline_ && baseline_nets_.size() != player_nets_.size()) {
      throw std::invalid_argument(
          "Baseline networks should be of the same size as player networks");
    }
    if (!spec_.game->GetType().provides_information_state_tensor) {
      throw std::invalid_argument(
          fmt::format("Game: {} does not provide information state tensor",
                      spec_.game->GetType().long_name));
    }
    if (spec_.game->GetType().chance_mode ==
        open_spiel::GameType::ChanceMode::kSampledStochastic) {
      throw std::invalid_argument("Solver only accepts deterministic or "
                                  "explicit chance modes for games.");
    }

    if (spec_.normalize_returns) {
      returns_normalizer_ = spec_.game->MaxUtility();
    } else {
      returns_normalizer_ = 1.0;
    }
  }

  SolverState init() const {
    SolverState state{AvgTabularPolicy(*spec_.game)};
    auto player_opt_config = torch::optim::SGDOptions(spec_.player_lr_schedule(0))
                          .weight_decay(spec_.weight_decay);
    for (const NetPtr &player_net : player_nets_) {
      state.player_opts.emplace_back(player_net->parameters(), player_opt_config);
    }
    if (has_baseline_) {
      auto baseline_opt_config = torch::optim::SGDOptions(spec_.baseline_lr_schedule(0))
                                   .weight_decay(spec_.weight_decay);
      for (const NetPtr &baseline_net : baseline_nets_) {
        state.baseline_opts.emplace_back(baseline_net->parameters(),
                                         baseline_opt_config);
      }
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

    for (open_spiel::Player player{0}; player < spec_.game->NumPlayers();
         player++) {

      if (state.step % spec_.player_update_freq == 0) {
        std::vector<ValueExample> strategy_memory_buffer;
        std::vector<UtilityExample> utility_memory_buffer;

        // collect player traversals
        for (uint32_t traversal = 0; traversal < spec_.player_traversals;
             traversal++) {
          auto init_state = spec_.game->NewInitialState();
          // no gradients required during traversals
          torch::NoGradGuard no_grad;

          traverse(player, *init_state, 1.0, 1.0, 1.0, strategy_memory_buffer,
                   utility_memory_buffer, state,
                   /*update_avg_strategy=*/update_tabular_policy_);
        }

        // update strategy network
        update_player(player, {std::move(strategy_memory_buffer)}, state);
      }

      if (has_baseline_) {
        std::vector<ValueExample> strategy_memory_buffer;
        std::vector<UtilityExample> utility_memory_buffer;

        // collect baseline traversals
        for (uint32_t traversal = 0; traversal < spec_.baseline_traversals;
             traversal++) {
          auto init_state = spec_.game->NewInitialState();
          // no gradients required during traversals
          torch::NoGradGuard no_grad;

          traverse(player, *init_state, 1.0, 1.0, 1.0, strategy_memory_buffer,
                   utility_memory_buffer, state, /*update_avg_strategy=*/false);
          }

        update_baseline(player, {std::move(utility_memory_buffer)}, state);
      }
    }
    state.step++;
  }

private:
  double traverse(open_spiel::Player player, open_spiel::State &state,
                  double player_reach_prob, double others_reach_prob,
                  double sample_reach_prob,
                  std::vector<ValueExample> &strategy_memory_buffer,
                  std::vector<UtilityExample> &utility_memory_buffer,
                  SolverState &solver_state,
                  bool update_avg_strategy) {
    if (state.IsTerminal()) {
      return state.PlayerReturn(player) / returns_normalizer_;
    } else if (state.IsChanceNode()) {
      auto chance_outcomes = state.ChanceOutcomes();
      const auto [chosen_action, sample_action_prob] =
          open_spiel::SampleAction(chance_outcomes, rng_);
      state.ApplyAction(chosen_action);
      return traverse(player, state, player_reach_prob,
                      others_reach_prob * sample_action_prob,
                      sample_reach_prob * sample_action_prob,
                      strategy_memory_buffer, utility_memory_buffer,
                      solver_state, update_avg_strategy);
    }
    SPIEL_CHECK_PROB(sample_reach_prob);

    const open_spiel::Player current_player = state.CurrentPlayer();
    // get information state representation for the current player
    const FeaturesType player_features = features_builder_.build(
        state.InformationStateTensor(current_player), spec_.device);

    const auto legal_actions = state.LegalActions(current_player);
    const auto legal_actions_mask =
        torch::tensor(state.LegalActionsMask(current_player), torch::kDouble);

    // calculate strategy and sampling probabilities
    Net &player_net = *player_nets_[current_player];
    auto logits = legal_actions_mask *
                  player_net.forward(features_builder_.batch(player_features))
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
    if (current_player == player && update_avg_strategy) {
      solver_state.avg_policy.UpdateStatePolicy(
          state.InformationStateString(player), strategy_probs,
          player_reach_prob, sample_reach_prob);
    }

    // NOTE: apply action to the state - it's modified so it should not be used
    // again in this function
    state.ApplyAction(chosen_action);
    // recursive call
    const double subgame_utility = traverse(
        player, state,
        player == current_player ? player_reach_prob * strategy_action_prob
                                 : player_reach_prob,
        player != current_player ? others_reach_prob * strategy_action_prob
                                 : others_reach_prob,
        sample_reach_prob * sample_action_prob,
        strategy_memory_buffer,
        utility_memory_buffer, solver_state, update_avg_strategy);

    // update utility memory buffer
    if (current_player == player && has_baseline_) {
      utility_memory_buffer.emplace_back(
          std::make_tuple(player_features.to(torch::kCPU), chosen_action),
          subgame_utility);
    }

    // caculate estimated baselines
    const torch::Tensor exp_utilities = [&]() {
      if (has_baseline_ && current_player == player && spec_.baseline_start_step <= solver_state.step) {
          Net &baseline_net = *baseline_nets_[player];
          return legal_actions_mask *
                 baseline_net.forward(features_builder_.batch(player_features))
                     .reshape({-1})
                     .toType(torch::kDouble)
                     .to(torch::kCPU);
      } else {
        return torch::zeros({spec_.game->NumDistinctActions()}, torch::kDouble);
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
    if (current_player == player) {
      torch::Tensor sampled_value =
          others_reach_prob * exp_utilities / sample_reach_prob;
      // update strategy memory buffer
      strategy_memory_buffer.emplace_back(player_features.to(torch::kCPU),
                                          sampled_value.to(torch::kCPU));
    }

    return state_utility;
  }

  void update_player(open_spiel::Player player,
                     data::VectorDataset<ValueExample> strategy_dataset,
                     SolverState &state) {
    auto data_loader = torch::data::make_data_loader(
        std::move(strategy_dataset),
        torch::data::DataLoaderOptions(spec_.batch_size));

    auto &optimizer = state.player_opts[player];
    // set learning rate
    torch::optim::SGDOptions &opt_config =
        dynamic_cast<torch::optim::SGDOptions &>(optimizer.defaults());
    opt_config.lr(spec_.player_lr_schedule(state.step));

    Net &player_net = *player_nets_[player];
    player_net.train();

    std::vector<FeaturesType> features_data;
    features_data.reserve(spec_.batch_size);
    std::vector<torch::Tensor> values_data;
    values_data.reserve(spec_.batch_size);

    double cumulative_loss = 0.0;
    double batch_count = 0.0;

    torch::Tensor zero = torch::zeros(
        {}, torch::TensorOptions().dtype(torch::kFloat32).device(spec_.device));

    for (const auto &data_batch : *data_loader) {
      for (const auto &example : data_batch) {
        features_data.push_back(example.data);
        values_data.push_back(example.target);
      }
      optimizer.zero_grad();
      FeaturesType features =
          features_builder_.batch(features_data).to(spec_.device);
      torch::Tensor values =
          spec_.eta *
          torch::stack(values_data).toType(torch::kFloat32).to(spec_.device);

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
      torch::Tensor entropy = -(torch::softmax(logits, 1) * torch::log_softmax(logits, 1)).mean();

      torch::Tensor loss = policy_loss - spec_.entropy_cost * entropy;
      loss.backward();
      // TODO Make it a parameter
      // torch::nn::utils::clip_grad_norm_(player_net.parameters(), 10.0);
      optimizer.step();

      cumulative_loss += loss.item<double>();
      batch_count += 1.0;
    }

    player_net.eval();
  }

  void update_baseline(open_spiel::Player player,
                       data::VectorDataset<UtilityExample> utility_dataset,
                       SolverState &state) {
    auto data_loader = torch::data::make_data_loader(
        std::move(utility_dataset),
        torch::data::DataLoaderOptions(spec_.batch_size));

    auto &optimizer = state.baseline_opts[player];
    // set learning rate
    torch::optim::SGDOptions &opt_config =
        dynamic_cast<torch::optim::SGDOptions &>(optimizer.defaults());
    opt_config.lr(spec_.baseline_lr_schedule(state.step));

    Net &baseline_net = *baseline_nets_[player];
    baseline_net.train();

    std::vector<FeaturesType> features_data;
    features_data.reserve(spec_.batch_size);
    std::vector<open_spiel::Action> actions_data;
    actions_data.reserve(spec_.batch_size);
    std::vector<double> utility_data;
    utility_data.reserve(spec_.batch_size);

    double cumulative_loss = 0.0;
    double batch_count = 0.0;

    for (const auto &data_batch : *data_loader) {
      // stack features and action indices
      for (const auto &example : data_batch) {
        features_data.push_back(std::get<0>(example.data));
        actions_data.push_back(std::get<1>(example.data));
        utility_data.push_back(example.target);
      }
      optimizer.zero_grad();

      FeaturesType features =
          features_builder_.batch(features_data).to(spec_.device);
      torch::Tensor batch_inds =
          torch::arange((int64_t)features_data.size(), spec_.device);
      torch::Tensor action_inds = torch::tensor(
          actions_data,
          torch::TensorOptions().dtype(torch::kInt64).device(spec_.device));

      torch::Tensor pred_utilities =
          baseline_net.forward(features).index({batch_inds, action_inds});
      torch::Tensor target_utilities = torch::tensor(
          utility_data,
          torch::TensorOptions().dtype(torch::kFloat32).device(spec_.device));

      torch::Tensor loss = torch::mse_loss(pred_utilities, target_utilities);
      loss.backward();
      optimizer.step();

      features_data.clear();
      actions_data.clear();
      utility_data.clear();

      cumulative_loss += loss.item<double>();
      batch_count += 1;
    }

    baseline_net.eval();
  }

private:
  const bool has_baseline_;
  double returns_normalizer_;
  const SolverSpec spec_;
  std::vector<NetPtr> player_nets_;
  std::vector<NetPtr> baseline_nets_;
  FeaturesBuilder features_builder_;
  std::mt19937 rng_;
  const bool update_tabular_policy_;
};

} // namespace dmc
