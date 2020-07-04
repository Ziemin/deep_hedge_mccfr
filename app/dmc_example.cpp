#include <dmc/deep_mw_cfr.hpp>
#include <dmc/features.hpp>
#include <dmc/nets.hpp>
#include <dmc/policy.hpp>
#include <fmt/core.h>
#include <memory>
#include <open_spiel/algorithms/tabular_exploitability.h>
#include <open_spiel/spiel.h>
#include <open_spiel/game_transforms/turn_based_simultaneous_game.h>
#include <unistd.h>
#include <vector>
#include <range/v3/all.hpp>

#include <torch/torch.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

ABSL_FLAG(std::string, game, "kuhn_poker", "The name of the game to play.");
ABSL_FLAG(int, players, 2, "How many players in this game, 0 for default.");
ABSL_FLAG(int, seed, 0, "Seed for the random number generator. 0 for auto.");
ABSL_FLAG(int, traversals, 10,
          "Number of individual player traversals before the policy updates");
ABSL_FLAG(int, baseline_traversals, 10,
          "Number of individual player traversals before the baseline updates");
ABSL_FLAG(int, iter, 1000, "Number of algorithm iterations");
ABSL_FLAG(double, eta, 1.0, "Eta parameters");
ABSL_FLAG(double, epsi, 0.1, "Epsilon for sampling policy");
ABSL_FLAG(bool, use_mw_update, false, "Use multiplicative weights update");
ABSL_FLAG(double, lr, 1e-3, "Learning rate");
ABSL_FLAG(bool, without_baseline, false, "Do not use baseline");
ABSL_FLAG(double, baseline_lr, 0.0, "Baseline Learning rate");
ABSL_FLAG(int, baseline_start, 0, "Step to start using baseline");
ABSL_FLAG(int, batch_size, 64, "Batch size");
ABSL_FLAG(double, wd, 1e-2, "Weight Decays");
ABSL_FLAG(std::vector<std::string>, units,
          std::vector<std::string>({"32", "32"}), "Networks hidden units");
ABSL_FLAG(std::vector<std::string>, baseline_units,
          std::vector<std::string>(), "Baseline Networks hidden units");
ABSL_FLAG(double, threshold, 2.0, "Logits threshold cut-off");
ABSL_FLAG(double, entropy_cost, 0.0, "Additional entropy loss for logits");
ABSL_FLAG(bool, normalize_returns, true, "Normalize player returns to range [-1, 1]");
ABSL_FLAG(int, player_update_freq, 1, "Number of steps player strategy network is updated");
ABSL_FLAG(bool, on_cpu, false, "Use only cpu");
ABSL_FLAG(int, eval_freq, 100, "Strategy evaluation frequency");

std::vector<uint32_t> get_units(const std::vector<std::string>& units_str) {
  return units_str
    | ranges::views::transform([](const std::string &num_str) {
                                 return (uint32_t)std::stoi(num_str);
                               })
    | ranges::to_vector;
}


using NetType = dmc::nets::StackedLinearNet;


int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string game_name = absl::GetFlag(FLAGS_game);
  const int p_count = absl::GetFlag(FLAGS_players);
  const int num_iterations = absl::GetFlag(FLAGS_iter);
  int seed = absl::GetFlag(FLAGS_seed);
  seed = seed == 0 ? time(0) : seed;
  const int eval_freq = absl::GetFlag(FLAGS_eval_freq);

  open_spiel::GameParameters params;
  params["players"] = open_spiel::GameParameter(p_count);
  if (game_name == "goofspiel") {
    params["num_cards"] = open_spiel::GameParameter(8);
  }

  auto game_candidate = open_spiel::LoadGame(game_name, params);
  auto game = game_candidate;
  if (game_candidate->GetType().dynamics == open_spiel::GameType::Dynamics::kSimultaneous) {
    game = open_spiel::ConvertToTurnBased(*game_candidate);
  }

  auto actions_size = game->NumDistinctActions();
  auto features_size = game->InformationStateTensorSize();

  torch::Device device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available() && !absl::GetFlag(FLAGS_on_cpu))
    device = torch::Device(torch::kCUDA);

  std::vector<uint32_t> hidden_units = get_units(absl::GetFlag(FLAGS_units));
  std::vector<uint32_t> baseline_units = get_units(absl::GetFlag(FLAGS_baseline_units));
  if (baseline_units.empty()) {
    baseline_units = hidden_units;
  }

  std::vector<std::shared_ptr<NetType>> players;
  std::vector<std::shared_ptr<NetType>> baselines;
  for (int p = 0; p < p_count; p++) {
    auto player_net = std::make_shared<dmc::nets::StackedLinearNet>(
        features_size, actions_size, hidden_units, true);
    player_net->to(device);
    players.push_back(std::move(player_net));

    if (!absl::GetFlag(FLAGS_without_baseline)) {
      auto baseline_net = std::make_shared<NetType>(
          features_size, actions_size, baseline_units, true);
      baseline_net->to(device);
      baselines.push_back(std::move(baseline_net));
    }
  }

  dmc::SolverSpec spec{game, device};
  spec.seed = seed;
  spec.epsilon = absl::GetFlag(FLAGS_epsi);
  spec.eta = absl::GetFlag(FLAGS_eta);
  spec.player_traversals = absl::GetFlag(FLAGS_traversals);
  spec.baseline_traversals = absl::GetFlag(FLAGS_baseline_traversals);
  spec.player_lr_schedule = [](auto step) { return absl::GetFlag(FLAGS_lr); };
  double baseline_lr = absl::GetFlag(FLAGS_baseline_lr);
  if (baseline_lr  != 0.0) {
    spec.baseline_lr_schedule = [](auto step) {
      return absl::GetFlag(FLAGS_lr);
    };
  } else {
    spec.baseline_lr_schedule = spec.player_lr_schedule;
  }
  spec.baseline_start_step = absl::GetFlag(FLAGS_baseline_start);
  spec.update_method = absl::GetFlag(FLAGS_use_mw_update)
                           ? dmc::UpdateMethod::MULTIPLICATIVE_WEIGHTS
                           : dmc::UpdateMethod::HEDGE;
  spec.weight_decay = absl::GetFlag(FLAGS_wd);
  spec.logits_threshold = absl::GetFlag(FLAGS_threshold);
  spec.entropy_cost = absl::GetFlag(FLAGS_entropy_cost);
  spec.normalize_returns = absl::GetFlag(FLAGS_normalize_returns);
  spec.player_update_freq = absl::GetFlag(FLAGS_player_update_freq);

  // create policy based on the latest neural network values
  dmc::NeuralPolicy neural_policy(players, dmc::features::RawInfoStateBuilder(), spec.device);

  // solver creation
  dmc::DeepMwCfrSolver solver(std::move(spec), std::move(players),
                              dmc::features::RawInfoStateBuilder(),
                              std::move(baselines));

  // training iterations
  auto state = solver.init();
  for (int i = 0; i < num_iterations; i++) {
    solver.run_iteration(state);
    if (i % eval_freq == 0) {
      const double avg_exploitability =
          open_spiel::algorithms::Exploitability(*game, state.avg_policy);
      const double last_exploitability =
          open_spiel::algorithms::Exploitability(*game, neural_policy);
      fmt::print("Iteration {}: Exploitability: Avg = {}, Last = {}\n",
                 i, avg_exploitability, last_exploitability);
    }
  }

  return 0;
}
