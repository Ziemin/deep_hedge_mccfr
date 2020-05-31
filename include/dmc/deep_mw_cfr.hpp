#pragma once

#include <dmc/baseline.hpp>
#include <dmc/player.hpp>
#include <functional>
#include <open_spiel/spiel.h>
#include <torch/torch.h>
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

  open_spiel::Game &game;
  torch::optim::Optimizer &opt;
  LRSchedule lr_schedule = DEFAULT_LR_SCHEDULE;
  double epsilon = DEFAULT_EPSILON;
  uint32_t player_traversals = DEFAULT_PLAYER_TRAVERSALS;
};

class DeepMwCfrSolver {

public:
  DeepMwCfrSolver(SolverSpec spec, std::vector<PlayerNet>& player_nets);
};

} // namespace dmc
