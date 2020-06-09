#pragma once

#include <dmc/baseline.hpp>
#include <dmc/player.hpp>
#include <functional>
#include <memory>
#include <open_spiel/spiel.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace dmc {

torch::Tensor some_fun();

using LRSchedule = std::function<double(uint64_t)>;

struct SolverSpec {
  static inline constexpr double DEFAULT_EPSILON = 0.1;
  static inline constexpr double DEFAULT_PLAYER_TRAVERSALS = 1;
  static inline constexpr double DEFAULT_LR_SCHEDULE(uint64_t /*step*/) {
    return 1e-4;
  }

  std::shared_ptr<const open_spiel::Game> game;
  LRSchedule lr_schedule = DEFAULT_LR_SCHEDULE;
  double epsilon = DEFAULT_EPSILON;
  uint32_t player_traversals = DEFAULT_PLAYER_TRAVERSALS;
};

struct SolverState {
  uint64_t step = 0;
};

struct NoBaseline {
  using FeatureType = void;
};

template <typename PlayerNet, typename FeaturesBuilder,
          typename BaselineNet = NoBaseline>
class DeepMwCfrSolver {
  using FeatureType = typename PlayerNet::FeatureType;

  using ValueExample =
      torch::data::Example<std::tuple<FeatureType, torch::Tensor>>;
  using UtilityExample = torch::data::Example<
      std::tuple<FeatureType, open_spiel::Action, torch::Tensor>>;

public:
  using PlayerNetPtr = std::shared_ptr<PlayerNet>;
  using BaselineNetPtr = std::shared_ptr<BaselineNet>;

  static_assert(std::is_same<BaselineNet, NoBaseline>() ||
                    std::is_same<typename PlayerNet::FeatureType,
                                 typename BaselineNet::FeatureType>(),
                "Players and Baselines feature tyeps should be the same");

  DeepMwCfrSolver(SolverSpec spec, std::vector<PlayerNetPtr> player_nets,
                  FeaturesBuilder features_builder,
                  std::vector<BaselineNetPtr> baseline_nets = {})
      : spec_(std::move(spec)), player_nets_(std::move(player_nets)),
        baseline_nets_(std::move(baseline_nets)),
        features_builder_(std::move(features_builder)) {

    if (player_nets_.size() != spec_.game->NumPlayers()) {
      throw std::invalid_argument(
          "There should be the a network for every player");
    }
    if (!baseline_nets_.empty() &&
        baseline_nets_.size() != player_nets_.size()) {
      throw std::invalid_argument(
          "Baseline networks should be either empty or contain the same number "
          "of elements as player networks.");
    }
  }

  SolverState init() const { return {0}; }

  void run_iteration(SolverState &state) {
    std::vector<ValueExample> strategy_memory_buffer;
    std::vector<UtilityExample> utility_memory_buffer;

    for (open_spiel::Player player{0}; player < spec_.game->NumPlayers();
         player++) {
      for (uint32_t traversal = 0; traversal < spec_.player_traversals;
           traversal++) {
        auto init_state = spec_.game->NewInitialState();
        traverse(player, *init_state, 1.0, 1.0, strategy_memory_buffer, utility_memory_buffer);
      }
    }
    state.step++;
  }

private:
  double traverse(open_spiel::Player player, open_spiel::State &state,
                  double player_reach_prob,
                  double others_reach_prob,
                  std::vector<ValueExample> &strategy_memory_buffer,
                  std::vector<UtilityExample> &utility_memory_buffer)
  {
    if (state.IsTerminal()) {
      return state.PlayerReturn(player);
    }
    if (state.IsChanceNode()) {
    } else {
    }

    return 0.0;
  }

private:
  const SolverSpec spec_;
  std::vector<PlayerNetPtr> player_nets_;
  std::vector<BaselineNetPtr> baseline_nets_;
  FeaturesBuilder features_builder_;
};

} // namespace dmc
