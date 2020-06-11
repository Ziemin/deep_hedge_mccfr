#include <dmc/deep_mw_cfr.hpp>
#include <memory>
#include <open_spiel/spiel.h>
#include <vector>
#include <unistd.h>

#include <torch/torch.h>

struct SimpleNet : torch::nn::Module {
  using FeaturesType = torch::Tensor;

  SimpleNet(uint32_t features_size, uint32_t actions_size)
      : fc_1(features_size, 30), fc_2(30, 60),
        fc_out(60, actions_size) {
    register_module("fc_1", fc_1);
    register_module("fc_2", fc_2);
    register_module("fc_out", fc_out);
  }

  torch::Tensor forward(torch::Tensor features) {
    auto out = torch::relu(fc_1(features));
    out = torch::relu(fc_2(out));
    return fc_out(out);
  }

private:
  torch::nn::Linear fc_1, fc_2, fc_out;
};

struct FeaturesBuilder {

  torch::Tensor build(const std::vector<double> &observation_tensor, const torch::Device& device) {
    return torch::tensor(observation_tensor).to(device);
  }

  torch::Tensor batch(std::vector<torch::Tensor> examples) {
    return torch::stack(examples, 0);
  }

  torch::Tensor batch(torch::Tensor example) {
    using namespace torch::indexing;
    return example.index({None, Slice{}});
  }
};


int main() {

  open_spiel::GameParameters params;
  auto game_name = "leduc_poker";
  int p_count = 2;
  params["players"] = open_spiel::GameParameter(p_count);

  auto game = open_spiel::LoadGame(game_name, params);
  auto actions_size = game->NumDistinctActions();
  auto features_size = game->ObservationTensorSize();

  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);

  std::vector<std::shared_ptr<SimpleNet>> players;
  std::vector<std::shared_ptr<SimpleNet>> baselines;
  for (int p = 0; p < p_count; p++) {
    auto player_net = std::make_shared<SimpleNet>(features_size, actions_size);
    player_net->to(device);
    players.push_back(std::move(player_net));

    auto baseline_net = std::make_shared<SimpleNet>(features_size, actions_size);
    baseline_net->to(device);
    baselines.push_back(std::move(baseline_net));
  }

  dmc::SolverSpec spec{game, device};
  spec.seed = time(0);
  spec.eta = 1.0;

  dmc::DeepMwCfrSolver solver(std::move(spec), std::move(players),
                              FeaturesBuilder(), std::move(baselines));

  auto state = solver.init();
  for (int i = 0; i < 100000; i++)
    solver.run_iteration(state);

  return 0;
}
