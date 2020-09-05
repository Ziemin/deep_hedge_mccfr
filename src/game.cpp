
#include <memory>
#include <nlohmann/json.hpp>
#include <open_spiel/spiel.h>
#include <dhc/game.hpp>
#include <open_spiel/game_transforms/turn_based_simultaneous_game.h>


namespace dhc::game {

  std::shared_ptr<const open_spiel::Game> load_game(const nlohmann::json &game_json) {
    std::string name = game_json["name"].get<std::string>();

    open_spiel::GameParameters params;
    if (game_json.contains("params")) {
      for (const auto& [name, value] : game_json["params"].items()) {
        if (value.is_string()) {
          params[name] = open_spiel::GameParameter(value.get<std::string>());
        } else if (value.is_number()) {
          params[name] = open_spiel::GameParameter(value.get<int>());
        }
      }
    }

    auto game = open_spiel::LoadGame(name, params);

    if (game->GetType().dynamics ==
        open_spiel::GameType::Dynamics::kSimultaneous) {
      game = open_spiel::ConvertToTurnBased(*game);
    }

    return game;
  }

} // namespace dhc::game
