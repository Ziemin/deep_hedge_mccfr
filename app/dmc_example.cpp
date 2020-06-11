#include <dmc/deep_mw_cfr.hpp>
#include <memory>
#include <open_spiel/spiel.h>
#include <unistd.h>
#include <vector>

#include <torch/torch.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

ABSL_FLAG(std::string, game, "kuhn_poker", "The name of the game to play.");
ABSL_FLAG(int, players, 2, "How many players in this game, 0 for default.");
ABSL_FLAG(int, seed, 0, "Seed for the random number generator. 0 for auto.");
ABSL_FLAG(int, traversals, 10,
          "Number of individual player traversals before the updates");
ABSL_FLAG(int, iter, 1000, "Number of algorithm iterations");
ABSL_FLAG(double, eta, 1.0, "Eta parameters");
ABSL_FLAG(bool, use_mw_update, false, "Use multiplicative weights update");

struct SimpleNet : torch::nn::Module {
  using FeaturesType = torch::Tensor;

  SimpleNet(uint32_t features_size, uint32_t actions_size)
      : fc_1(features_size, 32), fc_2(32, 48), fc_3(48, 64),
        fc_out(64, actions_size) {
    register_module("fc_1", fc_1);
    register_module("fc_2", fc_2);
    register_module("fc_3", fc_3);
    register_module("fc_out", fc_out);
  }

  torch::Tensor forward(torch::Tensor features) {
    auto out = torch::relu(fc_1(features));
    out = torch::relu(fc_2(out));
    out = torch::relu(fc_3(out));
    return fc_out(out);
  }

private:
  torch::nn::Linear fc_1, fc_2, fc_3, fc_out;
};

struct FeaturesBuilder {

  torch::Tensor build(const std::vector<double> &observation_tensor,
                      const torch::Device &device) {
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

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string game_name = absl::GetFlag(FLAGS_game);
  const int p_count = absl::GetFlag(FLAGS_players);
  const int num_iterations = absl::GetFlag(FLAGS_iter);
  int seed = absl::GetFlag(FLAGS_seed);
  seed = seed == 0 ? time(0) : seed;

  open_spiel::GameParameters params;
  params["players"] = open_spiel::GameParameter(p_count);

  auto game = open_spiel::LoadGame(game_name, params);
  auto actions_size = game->NumDistinctActions();
  auto features_size = game->ObservationTensorSize();

  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available())
    device = torch::Device(torch::kCUDA);

  std::vector<std::shared_ptr<SimpleNet>> players;
  std::vector<std::shared_ptr<SimpleNet>> baselines;
  for (int p = 0; p < p_count; p++) {
    auto player_net = std::make_shared<SimpleNet>(features_size, actions_size);
    player_net->to(device);
    players.push_back(std::move(player_net));

    auto baseline_net =
        std::make_shared<SimpleNet>(features_size, actions_size);
    baseline_net->to(device);
    baselines.push_back(std::move(baseline_net));
  }

  dmc::SolverSpec spec{game, device};
  spec.seed = seed;
  spec.eta = absl::GetFlag(FLAGS_eta);
  spec.player_traversals = absl::GetFlag(FLAGS_traversals);
  spec.lr_schedule = [](auto step) { return 1e-2; };
  spec.update_method = absl::GetFlag(FLAGS_use_mw_update)
                           ? dmc::UpdateMethod::MULTIPLICATIVE_WEIGHTS
                           : dmc::UpdateMethod::HEDGE;

  dmc::DeepMwCfrSolver solver(std::move(spec), std::move(players),
                              FeaturesBuilder(), std::move(baselines));

  auto state = solver.init();
  for (int i = 0; i < num_iterations; i++)
    solver.run_iteration(state);

  return 0;
}
