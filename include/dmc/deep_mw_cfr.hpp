#pragma once

#include <dmc/baseline.hpp>
#include <dmc/datasets.hpp>
#include <dmc/player.hpp>
#include <dmc/utils.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <functional>
#include <memory>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
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
  static inline constexpr double DEFAULT_LR_SCHEDULE(uint64_t /*step*/) {
    return 1e-3;
  }
  static inline constexpr UpdateMethod DEFAULT_UPDATE = UpdateMethod::HEDGE;
  static inline constexpr size_t DEFAULT_BATCH_SIZE = 64;

  std::shared_ptr<const open_spiel::Game> game;
  torch::Device device;
  LRSchedule lr_schedule = DEFAULT_LR_SCHEDULE;
  double epsilon = DEFAULT_EPSILON;
  double eta = DEFAULT_ETA;
  uint32_t player_traversals = DEFAULT_PLAYER_TRAVERSALS;
  UpdateMethod update_method = DEFAULT_UPDATE;
  size_t batch_size = DEFAULT_BATCH_SIZE;
  int seed = 0;
};

struct SolverState {
  uint64_t step = 0, time = 0;
  std::vector<torch::optim::SGD> player_opts;
  std::vector<torch::optim::SGD> baseline_opts;
  std::vector<open_spiel::TabularPolicy> avg_policies;
};

struct NoBaseline {
  using FeaturesType = void;
};

template <typename PlayerNet, typename FeaturesBuilder,
          typename BaselineNet = NoBaseline>
class DeepMwCfrSolver {
  using FeaturesType = typename PlayerNet::FeaturesType;

  using ValueExample = torch::data::Example<FeaturesType, torch::Tensor>;
  using UtilityExample =
      torch::data::Example<std::tuple<FeaturesType, open_spiel::Action>,
                           double>;

public:
  constexpr static bool has_baseline = !std::is_same<NoBaseline, BaselineNet>();
  using PlayerNetPtr = std::shared_ptr<PlayerNet>;
  using BaselineNetPtr = std::shared_ptr<BaselineNet>;

  static_assert(std::is_base_of<torch::nn::Module, PlayerNet>(),
                "PlayerNet has to be torch::nn::Module");
  static_assert(!has_baseline ||
                    std::is_base_of<torch::nn::Module, BaselineNet>(),
                "BaselineNet has to be torch::nn::Module");
  static_assert(!has_baseline ||
                    std::is_same<typename PlayerNet::FeaturesType,
                                 typename BaselineNet::FeaturesType>(),
                "Players and Baselines feature tyeps should be the same");

  DeepMwCfrSolver(SolverSpec spec, std::vector<PlayerNetPtr> player_nets,
                  FeaturesBuilder features_builder,
                  std::vector<BaselineNetPtr> baseline_nets = {},
                  bool update_tabular_policy = true)
      : spec_(std::move(spec)), player_nets_(std::move(player_nets)),
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
    if (has_baseline && baseline_nets_.size() != player_nets_.size()) {
      throw std::invalid_argument(
          "Baseline networks should be of the same size as player networks");
    }
    if (!spec_.game->GetType().provides_observation_tensor) {
      throw std::invalid_argument(
          fmt::format("Game: {} does not provide observation tensor",
                      spec_.game->GetType().long_name));
    }
    if (spec_.game->GetType().chance_mode ==
        open_spiel::GameType::ChanceMode::kSampledStochastic) {
      throw std::invalid_argument("Solver only accepts deterministic or "
                                  "explicit chance modes for games.");
    }
  }

  SolverState init() const {
    SolverState state;
    torch::optim::SGDOptions opt_config(spec_.lr_schedule(0));
    for (const PlayerNetPtr &player_net : player_nets_) {
      state.player_opts.emplace_back(player_net->parameters(), opt_config);
    }
    if constexpr (has_baseline) {
      for (const BaselineNetPtr &baseline_net : baseline_nets_) {
        state.baseline_opts.emplace_back(baseline_net->parameters(),
                                         opt_config);
      }
    }
    if (update_tabular_policy_) {
      for (open_spiel::Player p{0}; p < spec_.game->NumPlayers(); p++) {
        state.avg_policies.emplace_back(*spec_.game);
      }
    }

    return state;
  }

  void run_iteration(SolverState &state) {
    // set networks to eval mode
    for (PlayerNetPtr &player_ptr : player_nets_)
      player_ptr->eval();
    if constexpr (has_baseline)
      for (BaselineNetPtr &baseline_ptr : baseline_nets_)
        baseline_ptr->eval();

    const uint32_t start_time = state.time;
    for (open_spiel::Player player{0}; player < spec_.game->NumPlayers();
         player++) {

      std::vector<ValueExample> strategy_memory_buffer;
      std::vector<UtilityExample> utility_memory_buffer;

      // collect traversals
      for (uint32_t traversal = 0; traversal < spec_.player_traversals;
           traversal++) {
        state.time = start_time + traversal + 1;
        auto init_state = spec_.game->NewInitialState();
        // no gradients required during traversals
        torch::NoGradGuard no_grad;

        traverse(player, *init_state, 1.0, 1.0, strategy_memory_buffer,
                 utility_memory_buffer, state);
      }

      // update networks
      update_player(player, {std::move(strategy_memory_buffer)}, state);
      if constexpr (has_baseline) {
        update_baseline(player, {std::move(utility_memory_buffer)}, state);
      }
    }
    state.step++;
    state.time = start_time + spec_.player_traversals;
  }

private:
  double traverse(open_spiel::Player player, open_spiel::State &state,
                  double player_reach_prob, double others_reach_prob,
                  std::vector<ValueExample> &strategy_memory_buffer,
                  std::vector<UtilityExample> &utility_memory_buffer,
                  SolverState &solver_state) {
    if (state.IsTerminal()) {
      return state.PlayerReturn(player);
    }

    // get observation state representation
    const FeaturesType obs_features =
        features_builder_.build(state.ObservationTensor(player), spec_.device);
    const open_spiel::Player current_player = state.CurrentPlayer();
    const auto legal_actions = state.LegalActions(current_player);

    // calculate strategy and sampling probabilities
    const auto [strategy_probs, sampling_probs] = [&]() {
      if (current_player == open_spiel::kChancePlayerId) {
        auto chance_outcomes = state.ChanceOutcomes();
        auto probs = chance_outcomes |
                     ranges::views::transform(
                         [](auto act_prob) { return act_prob.second; }) |
                     ranges::to<std::vector>();
        return std::make_pair(probs, probs);
      } else {
        PlayerNet &player_net = *player_nets_[current_player];
        auto logits = player_net.forward(features_builder_.batch(obs_features))
                          .reshape({-1})
                          .toType(torch::kDouble)
                          .to(torch::kCPU);
        auto strategy_probs = utils::get_probabilities(legal_actions, logits);
        const double epsilon = spec_.epsilon;
        const double act_count = legal_actions.size();
        auto sampling_probs =
            epsilon / act_count + (1.0 - epsilon) * strategy_probs;

        return std::make_pair(utils::to_vector<double>(strategy_probs),
                              utils::to_vector<double>(sampling_probs));
      }
    }();

    // update average policy
    if (current_player == player && update_tabular_policy_) {
      auto &policy_table = solver_state.avg_policies[player].PolicyTable();
      auto &state_policy = policy_table[state.InformationStateString(player)];
      for (size_t act_ind = 0; act_ind < state_policy.size(); act_ind++) {
        auto [action, prev_prob] = state_policy[act_ind];
        double new_prob = strategy_probs[act_ind];
        if (solver_state.time > 1) {
          double time_d = solver_state.time;
          new_prob = (time_d - 1.0) * prev_prob / time_d + new_prob / time_d;
        }
        state_policy[act_ind] = {action, new_prob};
      }
    }

    const auto [chosen_action, sample_action_prob] = open_spiel::SampleAction(
        utils::to_actions_and_probs(legal_actions, sampling_probs), rng_);
    const uint32_t action_ind =
        ranges::distance(ranges::begin(legal_actions),
                         ranges::find(legal_actions, chosen_action));
    const double strategy_action_prob = strategy_probs[action_ind];

    // NOTE: apply action to the state - it's modified so it should not be used
    // again in this function
    state.ApplyAction(chosen_action);
    // recursive call
    const double subgame_utility = traverse(
        player, state,
        player == current_player ? player_reach_prob * sample_action_prob
                                 : player_reach_prob,
        player != current_player ? others_reach_prob * strategy_action_prob
                                 : others_reach_prob,
        strategy_memory_buffer, utility_memory_buffer, solver_state);

    // update utility memory buffer
    if (current_player != open_spiel::kChancePlayerId) {
      utility_memory_buffer.emplace_back(
          std::make_tuple(obs_features.to(torch::kCPU), chosen_action),
          subgame_utility);
    }

    // caculate estimated baselines
    const torch::Tensor exp_utilities = [&]() {
      // NOTE we're not using baselines for chance nodes, due to their different
      // action spaces we'd have to use a separate network for chance node
      // actions
      if constexpr (has_baseline) {
        if (current_player == open_spiel::kChancePlayerId) {
          return torch::zeros({spec_.game->MaxChanceOutcomes()},
                              torch::kDouble);
        } else {
          BaselineNet &baseline_net = *baseline_nets_[player];
          return baseline_net.forward(obs_features)
              .reshape({-1})
              .toType(torch::kDouble)
              .to(torch::kCPU);
        }
      } else {
        if (current_player == open_spiel::kChancePlayerId) {
          return torch::zeros({spec_.game->MaxChanceOutcomes()},
                              torch::kDouble);
        } else {
          return torch::zeros({spec_.game->NumDistinctActions()},
                              torch::kDouble);
        }
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
          others_reach_prob * exp_utilities / player_reach_prob;
      // update strategy memory buffer
      strategy_memory_buffer.emplace_back(obs_features.to(torch::kCPU),
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
    opt_config.lr(spec_.lr_schedule(state.step));

    PlayerNet &player_net = *player_nets_[player];
    player_net.train();

    std::vector<FeaturesType> features_data;
    features_data.reserve(spec_.batch_size);
    std::vector<torch::Tensor> values_data;
    values_data.reserve(spec_.batch_size);

    double cumulative_loss = 0.0;
    double batch_count = 0.0;

    for (const auto &data_batch : *data_loader) {
      for (const auto &example : data_batch) {
        features_data.push_back(example.data);
        values_data.push_back(example.target);
      }
      player_net.zero_grad();
      FeaturesType features =
          features_builder_.batch(features_data).to(spec_.device);
      torch::Tensor values =
          torch::stack(values_data).toType(torch::kFloat32).to(spec_.device);

      torch::Tensor logits = player_net.forward(features);

      torch::Tensor update_mult = spec_.eta * values;
      if (spec_.update_method == UpdateMethod::MULTIPLICATIVE_WEIGHTS) {
        update_mult = torch::log(1.0 + update_mult);
      }
      update_mult.requires_grad_(false);

      torch::Tensor loss = -update_mult.mm(logits.t()).mean();
      loss.backward();
      optimizer.step();

      cumulative_loss += loss.item<double>();
      batch_count += 1.0;
    }

    // fmt::print("Step {}, strategy loss for player {} = {}\n", state.step,
    //            player, cumulative_loss / batch_count);

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
    opt_config.lr(spec_.lr_schedule(state.step));

    BaselineNet &baseline_net = *baseline_nets_[player];
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
      baseline_net.zero_grad();

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
    // fmt::print("Step {}, baseline loss for player {} = {}\n", state.step,
               // player, cumulative_loss / batch_count);

    baseline_net.eval();
  }

private:
  const SolverSpec spec_;
  std::vector<PlayerNetPtr> player_nets_;
  std::vector<BaselineNetPtr> baseline_nets_;
  FeaturesBuilder features_builder_;
  std::mt19937 rng_;
  bool update_tabular_policy_;
};

} // namespace dmc
