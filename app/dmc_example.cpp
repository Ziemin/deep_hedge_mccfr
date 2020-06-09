#include <dmc/deep_mw_cfr.hpp>
#include <open_spiel/spiel.h>
#include <vector>
#include <memory>

#include <torch/torch.h>


struct DummyPlayerNet : torch::nn::Module {
  using FeatureType = torch::Tensor;
};

struct DummyBaselineNet : torch::nn::Module {
  using FeatureType = torch::Tensor;
};


int main() {

  std::vector<std::shared_ptr<DummyPlayerNet>> players{std::make_shared<DummyPlayerNet>(),
                                                       std::make_shared<DummyPlayerNet>()};
  std::vector<std::shared_ptr<DummyBaselineNet>> baselines {};

  open_spiel::GameParameters params;
  auto game_name = "kuhn_poker";
  params["players"] = open_spiel::GameParameter(2);

  auto game = open_spiel::LoadGame(game_name, params);
  dmc::SolverSpec spec{game};

  dmc::DeepMwCfrSolver solver(std::move(spec), std::move(players), 6, std::move(baselines));

  auto state = solver.init();
  solver.run_iteration(state);

  return 0;
}
