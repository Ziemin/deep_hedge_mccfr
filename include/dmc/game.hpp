#pragma once

#include <memory>
#include <open_spiel/spiel.h>
#include <nlohmann/json.hpp>

namespace dmc::game {

  std::shared_ptr<const open_spiel::Game> load_game(const nlohmann::json& game_json);

} // namespace dmc::game
