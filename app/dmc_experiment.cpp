#include <dmc/deep_mw_cfr.hpp>
#include <dmc/features.hpp>
#include <dmc/game.hpp>
#include <dmc/nets.hpp>
#include <dmc/policy.hpp>
#include <boost/filesystem.hpp>

#include <chrono>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <open_spiel/algorithms/tabular_exploitability.h>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>
#include <unistd.h>
#include <vector>

#include <torch/torch.h>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

using NetType = dmc::nets::StackedLinearNet;
using NetPtr = std::shared_ptr<NetType>;

namespace bfs = boost::filesystem;

struct Experiment {
  std::shared_ptr<const open_spiel::Game> game;
  dmc::SolverSpec spec;
  std::vector<NetPtr> player_nets, baseline_nets;
  torch::Device device;

  static Experiment from_json(const nlohmann::json &experiment_json) {
    // load game
    auto game = dmc::game::load_game(experiment_json["game"]);
    // load solver specification
    auto spec = dmc::SolverSpec::from_json(experiment_json["spec"]);

    // set training device
    torch::Device device = torch::Device(torch::kCPU);
    if (experiment_json.contains("device") &&
        experiment_json["device"].get<std::string>() == "cuda" &&
        torch::cuda::is_available()) {

      device = torch::Device(torch::kCUDA);
    }
    // load player networks
    auto actions_size = game->NumDistinctActions();
    auto features_size = game->InformationStateTensorSize();
    std::vector<uint32_t> player_units =
        experiment_json["networks"]["player_units"]
            .get<std::vector<uint32_t>>();
    std::vector<uint32_t> baseline_units =
        experiment_json["networks"]["baseline_units"]
            .get<std::vector<uint32_t>>();
    std::vector<std::shared_ptr<NetType>> players;
    std::vector<std::shared_ptr<NetType>> baselines;
    for (int p = 0; p < game->NumPlayers(); p++) {
      auto player_net = std::make_shared<dmc::nets::StackedLinearNet>(
          features_size, actions_size, player_units, true);
      player_net->to(device);
      players.push_back(std::move(player_net));

      if (!baseline_units.empty()) {
        auto baseline_net = std::make_shared<NetType>(
            features_size, actions_size, baseline_units, true);
        baseline_net->to(device);
        baselines.push_back(std::move(baseline_net));
      }
    }
    return Experiment{game, spec, players, baselines, device};
  }
};

ABSL_FLAG(std::string, config, "./config.json", "Path to config");
ABSL_FLAG(std::string, name, "", "Experiment name");
ABSL_FLAG(std::string, dir, "", "Experiment dir");
ABSL_FLAG(int, eval_freq, 1000, "Strategy evaluation frequency");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  std::string experiment_name = absl::GetFlag(FLAGS_name);
  bfs::path experiment_dir(absl::GetFlag(FLAGS_dir));
  std::time_t time = std::time(nullptr);

  // read experiment configuration
  nlohmann::json config_json;
  {
    std::ifstream in_stream(absl::GetFlag(FLAGS_config));
    in_stream >> config_json;
  }
  Experiment experiment = Experiment::from_json(config_json);
  // add defaults to config to see all the parameters
  config_json["spec"] = experiment.spec.to_json();

  if (!experiment_dir.empty()) {
    fmt::print("Before name\n");
    experiment_name = experiment_name.empty()
                          ? experiment.game->GetType().short_name
                          : experiment_name;
    experiment_dir /= fmt::format("{}/{:%Y-%m-%d_%H-%M-%S}", experiment_name, *std::localtime(&time));
    if (!bfs::exists(experiment_dir)) {
      bfs::create_directories(experiment_dir);
    }
    fmt::print("Will save experiment results to: {}\n", experiment_dir.string());
    std::ofstream config_f(experiment_dir / "config.json");
    config_f << std::setw(4) << config_json;
  }

  fmt::print("Running experiment for configuration:\n{}\n", config_json.dump(4));

  dmc::features::RawInfoStateBuilder features_builder;

  // instantiate solver
  dmc::DeepMwCfrSolver solver(experiment.game, experiment.spec,
                              experiment.player_nets, features_builder,
                              experiment.device, experiment.baseline_nets);

  // create policy based on the latest neural network values
  dmc::NeuralPolicy neural_policy(experiment.player_nets,
                                  dmc::features::RawInfoStateBuilder(),
                                  experiment.device);

  auto state = solver.init();
  const uint64_t eval_freq = absl::GetFlag(FLAGS_eval_freq);
  nlohmann::json stats;

  fmt::print("Running experiment...\n");

  while (state.step < experiment.spec.max_steps) {
    solver.run_iteration(state);
    if (state.step == 1 || state.step % eval_freq == 0) {
      time = std::time(nullptr);

      const double avg_exploitability = open_spiel::algorithms::Exploitability(
          *experiment.game, state.avg_policy);
      const double last_exploitability = open_spiel::algorithms::Exploitability(
          *experiment.game, neural_policy);

      nlohmann::json step_stats = {
          {"step", state.step},
          {"time", fmt::format("{:%Y-%m-%d_%H-%M-%S}", *std::localtime(&time))},
          {"avg_strategy_exploitability", avg_exploitability},
          {"final_strategy_exploitability", last_exploitability}};
      fmt::print("{}\n", step_stats.dump(2));
      stats.push_back(std::move(step_stats));

      // save stats
      std::ofstream stats_f(experiment_dir / "stats.json");
      stats_f << stats;
    }
  }

  return 0;
}
