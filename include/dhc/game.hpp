#pragma once

#include <memory>
#include <open_spiel/spiel.h>
#include <nlohmann/json.hpp>

namespace dhc::game {

  std::shared_ptr<const open_spiel::Game> load_game(const nlohmann::json& game_json);

} // namespace dhc::game
