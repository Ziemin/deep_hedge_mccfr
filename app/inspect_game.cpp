#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/random/uniform_int_distribution.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <open_spiel/spiel.h>
#include <string>
#include <vector>

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(int, players, 0, "How many players in this game, 0 for default.");
ABSL_FLAG(int, seed, 0, "Seed for the random number generator. 0 for auto.");


int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string game_name = absl::GetFlag(FLAGS_game);
  const int players = absl::GetFlag(FLAGS_players);
  const int seed = absl::GetFlag(FLAGS_seed);

  fmt::print("Creating a game..\n");
  open_spiel::GameParameters params;
  if (players > 0)
    params["players"] = open_spiel::GameParameter(players);

  auto game = open_spiel::LoadGame(game_name, params);
  if (!game) {
    fmt::print(stderr, "Problem loading game {}, exiting...", game_name);
    return -1;
  }

  return 0;
}

