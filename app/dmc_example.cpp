#include <dmc/deep_mw_cfr.hpp>
#include <dmc/features.hpp>
#include <dmc/nets.hpp>
#include <fmt/core.h>
#include <memory>
#include <open_spiel/algorithms/tabular_exploitability.h>
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
ABSL_FLAG(double, epsi, 0.1, "Epsilon for sampling policy");
ABSL_FLAG(bool, use_mw_update, false, "Use multiplicative weights update");
ABSL_FLAG(double, lr, 1e-3, "Learning rate");
ABSL_FLAG(double, wd, 1e-2, "Weight Decays");
ABSL_FLAG(int, units_factor, 4, "Unit layers factor");
ABSL_FLAG(double, threshold, 2.0, "Logits threshold cut-off");

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
  auto features_size = game->InformationStateTensorSize();

  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available())
    device = torch::Device(torch::kCUDA);

  const int units_factor = absl::GetFlag(FLAGS_units_factor);
  std::array<uint32_t, 3> hidden_units;
  hidden_units[0] = units_factor * 12;
  hidden_units[1] = units_factor * 12;
  hidden_units[2] = units_factor * 16;

  std::vector<std::shared_ptr<dmc::nets::StackedLinearNet>> players;
  std::vector<std::shared_ptr<dmc::nets::StackedLinearNet>> baselines;
  for (int p = 0; p < p_count; p++) {
    auto player_net = std::make_shared<dmc::nets::StackedLinearNet>(
        features_size, actions_size, hidden_units);
    player_net->to(device);
    players.push_back(std::move(player_net));

    auto baseline_net = std::make_shared<dmc::nets::StackedLinearNet>(
        features_size, actions_size, hidden_units);
    baseline_net->to(device);
    baselines.push_back(std::move(baseline_net));
  }

  dmc::SolverSpec spec{game, device};
  spec.seed = seed;
  spec.epsilon = absl::GetFlag(FLAGS_epsi);
  spec.eta = absl::GetFlag(FLAGS_eta);
  spec.player_traversals = absl::GetFlag(FLAGS_traversals);
  spec.lr_schedule = [](auto step) { return absl::GetFlag(FLAGS_lr); };
  spec.update_method = absl::GetFlag(FLAGS_use_mw_update)
                           ? dmc::UpdateMethod::MULTIPLICATIVE_WEIGHTS
                           : dmc::UpdateMethod::HEDGE;
  spec.weight_decay = absl::GetFlag(FLAGS_wd);
  spec.logits_threshold = absl::GetFlag(FLAGS_threshold);

  dmc::DeepMwCfrSolver solver(std::move(spec), std::move(players),
                              dmc::features::RawInfoStateBuilder());

  auto state = solver.init();
  for (int i = 0; i < num_iterations; i++) {
    solver.run_iteration(state);
    if (i % 10 == 0) {
      const double exploitability =
          open_spiel::algorithms::Exploitability(*game, state.avg_policy);
      fmt::print("Iteration {}: Exploitability = {}\n", i, exploitability);
    }
  }

  return 0;
}
