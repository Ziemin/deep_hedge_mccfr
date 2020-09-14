#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/random/uniform_int_distribution.h>
#include <dhc/format.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <open_spiel/game_parameters.h>
#include <open_spiel/game_transforms/turn_based_simultaneous_game.h>
#include <open_spiel/policy.h>
#include <open_spiel/spiel.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

ABSL_FLAG(std::string, game, "", "The name of the game to play.");
ABSL_FLAG(int, players, 0, "How many players in this game, 0 for default.");
ABSL_FLAG(int, seed, 0, "Seed for the random number generator. 0 for auto.");
ABSL_FLAG(bool, playthrough, false, "Generate random playthrough.");
ABSL_FLAG(bool, print_legal_actions, false, "Print all legal actions");
ABSL_FLAG(bool, print_string_reprs, false,
          "Print string state representations");
ABSL_FLAG(bool, print_tensors, false, "Print tensor state representatins");
ABSL_FLAG(bool, print_all_perspectives, false, "Print tensors and strings for every player");

void generate_playthrough(const open_spiel::Game &game, int seed,
                          bool print_legal_actions, bool print_strings,
                          bool print_tensors, bool print_all_perspectives) {
  std::mt19937 rng(seed);

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::string state_format_str =
      fmt::format("\n--- State: --------------------------\n{{:{}{}{}{}}}\n",
                  print_legal_actions ? "a" : "",
                  print_strings ? "s" : "",
                  print_tensors ? "t" : "",
                  print_all_perspectives ? "p" : "");

  while (!state->IsTerminal()) {
    fmt::print(state_format_str, *state);

    if (state->IsChanceNode()) {

      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      open_spiel::Action action = open_spiel::SampleAction(outcomes, rng).first;
      fmt::print("Sampled Outcome: {}\n",
                 state->ActionToString(open_spiel::kChancePlayerId, action));
      state->ApplyAction(action);

    } else if (state->IsSimultaneousNode()) {

      std::vector<open_spiel::Action> joint_action;
      for (auto player = open_spiel::Player{0}; player < game.NumPlayers();
           ++player) {
        std::vector<open_spiel::Action> actions = state->LegalActions(player);
        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        open_spiel::Action action = actions[dis(rng)];
        joint_action.push_back(action);
        fmt::format("Player {} chose {}\n", player,
                    state->ActionToString(player, action));
      }
      state->ApplyActions(joint_action);

    } else {
      int player = state->CurrentPlayer();
      std::vector<open_spiel::Action> actions = state->LegalActions();
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      auto action = actions[dis(rng)];
      fmt::print("Sampled Action: {}\n", state->ActionToString(player, action));
      state->ApplyAction(action);
    }
  }
  fmt::print("Final State:\n{}\n", *state);
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string game_name = absl::GetFlag(FLAGS_game);
  const int players = absl::GetFlag(FLAGS_players);
  const int seed = absl::GetFlag(FLAGS_seed);
  const bool gen_playthrough = absl::GetFlag(FLAGS_playthrough);

  if (game_name.empty()) {
    fmt::print("Please, provide a game name.\nAvailable games:\n- {}\n",
               fmt::join(open_spiel::RegisteredGames(), "\n- "));
    return -1;
  }

  fmt::print("Creating a game..\n");
  open_spiel::GameParameters params;
  if (players > 0)
    params["players"] = open_spiel::GameParameter(players);

  if (game_name == "goofspiel") {
    params["num_cards"] = open_spiel::GameParameter(4);
  }

  auto game = open_spiel::LoadGame(game_name, params);
  if (!game) {
    fmt::print(stderr, "Problem loading game {}, exiting...", game_name);
    return -1;
  }
  if (game->GetType().dynamics ==
      open_spiel::GameType::Dynamics::kSimultaneous) {
    game = open_spiel::ConvertToTurnBased(*game);
  }
  fmt::print("{}\n", *game);
  open_spiel::TabularPolicy policy(*game);
  fmt::print("Number of information sets: {}\n", policy.PolicyTable().size());

  if (gen_playthrough) {
    fmt::print("Generating random playthrough...\n");
    generate_playthrough(*game, seed == 0 ? time(0) : seed,
                         absl::GetFlag(FLAGS_print_legal_actions),
                         absl::GetFlag(FLAGS_print_string_reprs),
                         absl::GetFlag(FLAGS_print_tensors),
                         absl::GetFlag(FLAGS_print_all_perspectives));
  }

  return 0;
}
