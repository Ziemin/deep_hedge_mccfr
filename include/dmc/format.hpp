#pragma once
#include <fmt/format.h>
#include <open_spiel/game_parameters.h>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>

namespace fmt {
template <>
struct formatter<open_spiel::GameType::RewardModel>
    : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(open_spiel::GameType::RewardModel rm, FormatContext &ctx) {
    using RewardModel = open_spiel::GameType::RewardModel;
    std::string_view name =
        rm == RewardModel::kRewards ? "RL-Style" : "Game-Style";
    return formatter<std::string_view>::format(name, ctx);
  }
};

template <>
struct formatter<open_spiel::TensorLayout> : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(open_spiel::TensorLayout tl, FormatContext &ctx) {
    std::string_view name =
        tl == open_spiel::TensorLayout::kCHW ? "CHW" : "HWC";
    return formatter<std::string_view>::format(name, ctx);
  }
};

template <>
struct formatter<open_spiel::GameParameter::Type>
    : formatter<std::string_view> {
  template <typename FormatContext>
  auto format(open_spiel::GameParameter::Type type, FormatContext &ctx) {
    std::string_view name = "unknown";
    switch (type) {
    case open_spiel::GameParameter::Type::kUnset:
      name = "unset";
      break;
    case open_spiel::GameParameter::Type::kInt:
      name = "int";
      break;
    case open_spiel::GameParameter::Type::kDouble:
      name = "double";
      break;
    case open_spiel::GameParameter::Type::kString:
      name = "string";
      break;
    case open_spiel::GameParameter::Type::kBool:
      name = "bool";
      break;
    case open_spiel::GameParameter::Type::kGame:
      name = "Game";
      break;
    };
    return formatter<std::string_view>::format(name, ctx);
  }
};

template <> struct formatter<open_spiel::Game> {

  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const open_spiel::Game &game, FormatContext &ctx) {
    auto it = ctx.out();
    auto game_type = game.GetType();
    it = format_to(it, "Game: {} ({})\n", game_type.long_name,
                   game_type.short_name);
    it = format_to(it, "* Dynamics: {}\n", game_type.dynamics);
    it = format_to(it, "* Chance Mode: {}\n", game_type.chance_mode);
    it = format_to(it, "* Information: {}\n", game_type.information);
    it = format_to(it, "* Utility: {}\n", game_type.utility);
    it = format_to(it, "* Utility Range: [{}, {}]\n", game.MinUtility(),
                   game.MaxUtility());
    if (game_type.utility == open_spiel::GameType::Utility::kConstantSum) {
      it = format_to(it, "\t- Utility Sum: {}\n", game.UtilitySum());
    }
    it = format_to(it, "* Reward Model: {}\n", game_type.reward_model);
    it = format_to(it, "* Num Distinct Actions: {}\n",
                   game.NumDistinctActions());
    it = format_to(it, "* Policy Tensor Shape: {}\n", game.PolicyTensorShape());
    it = format_to(it, "* Max Chance Outcomes: {}\n", game.MaxChanceOutcomes());
    it = format_to(it, "* Num Players: {}\n", game.NumPlayers());
    it = format_to(it, "* Maximum Game Lenth (over player actions): {}\n",
                   game.MaxGameLength());

    it = format_to(it, "* Provides Information State String: {}\n",
                   game_type.provides_information_state_string);
    it = format_to(it, "* Provides Information State Tensor: {}\n",
                   game_type.provides_information_state_tensor);
    if (game_type.provides_information_state_tensor) {
      it = format_to(it, "\t- Tensor Shape: {}\n",
                     game.InformationStateTensorShape());
      it = format_to(it, "\t- Tensor Layout: {}\n",
                     game.InformationStateTensorLayout());
    }
    it = format_to(it, "* Provides Observation String: {}\n",
                   game_type.provides_observation_string);
    it = format_to(it, "* Provides Observation Tensor: {}\n",
                   game_type.provides_observation_string);
    if (game_type.provides_observation_tensor) {
      it = format_to(it, "\t- Tensor Shape: {}\n",
                     game.ObservationTensorShape());
      it = format_to(it, "\t- Tensor Layout: {}\n",
                     game.ObservationTensorLayout());
    }
    it = format_to(it, "* Game Params:\n");
    for (const auto &[name, param] : game_type.parameter_specification) {
      it = format_to(it, "\t- '{}': {} {}\n", name, param.type(),
                     param.is_mandatory() ? "(required)" : "");
    }
    return it;
  }
};

template <> struct formatter<open_spiel::State> {
  bool print_legal_actions = false;
  bool print_string_repr = false;
  bool print_tensor_repr = false;
  bool show_all_perspectives = false;

  constexpr auto parse(format_parse_context &ctx) {
    auto end = ctx.end();
    auto it = ctx.begin();
    while (it != end && *it != '}') {
      if (*it == 'a')
        print_legal_actions = true;
      else if (*it == 's')
        print_string_repr = true;
      else if (*it == 't')
        print_tensor_repr = true;
      else if (*it == 'p')
        show_all_perspectives = true;
      else
        throw fmt::format_error(
            fmt::format("Unrecognized format char: {}", *it));
      ++it;
    }
    if (it != end && *it != '}')
      throw fmt::format_error("Invalid format");
    return it;
  }

  template <typename FormatContext>
  auto format(const open_spiel::State &state, FormatContext &ctx) {
    auto it = ctx.out();
    auto game = state.GetGame();
    auto game_type = game->GetType();

    if (state.IsTerminal()) {
      it = format_to(it, "Type: Terminal");
      it = format_to(it, "Players Returns:\n{}\n", state.Returns());
      return it;
    }

    auto player = state.CurrentPlayer();
    if (state.IsChanceNode()) {
      it = format_to(it, "Type: Chance\n");
    } else if (state.IsSimultaneousNode()) {
      it = format_to(it, "Type: Simultaneous\n");

      for (auto player = open_spiel::Player{0}; player < game->NumPlayers(); player++) {
        if (print_string_repr or print_tensor_repr)
          it = format_to(it, "Player {} perspective:\n", player);
        if (print_string_repr) {
          if (game_type.provides_information_state_string)
            it = format_to(it, "Info State String:\n{}\n",
                           player, state.InformationStateString(player));
          if (game_type.provides_observation_string)
            it = format_to(it, "Obervation String:\n{}\n",
                           player, state.ObservationString(player));
        }
        if (print_tensor_repr) {
          if (game_type.provides_information_state_tensor)
            it = format_to(it, "Info State Tensor:\n{}\n",
                           state.InformationStateTensor(player));
          if (game_type.provides_observation_tensor)
            it = format_to(it, "Observation Tensor:\n{}\n",
                           state.ObservationTensor(player));
        }
      }

    } else {
      it = format_to(it, "Type: Decision\n");
      it = format_to(it, "Current Player: {}\n", player);

      if (print_string_repr) {
        if (game_type.provides_information_state_string) {
          it = format_to(it, "Info State String For Current Player:\n{}\n",
                         state.InformationStateString(player));
          if (show_all_perspectives) {
            for (auto other_player = open_spiel::Player{0}; other_player < game->NumPlayers(); other_player++) {
              if (other_player == player) continue;
              it = format_to(it, "Info State String For Player {}:\n{}\n",
                             other_player,
                             state.InformationStateString(other_player));
            }
          }
        }
        if (game_type.provides_observation_string) {
          it = format_to(it, "Obervation String For Current Player:\n{}\n",
                         state.ObservationString(player));
          if (show_all_perspectives) {
            for (auto other_player = open_spiel::Player{0};
                 other_player < game->NumPlayers(); other_player++) {
              if (other_player == player)
                continue;
              it = format_to(it, "Observation String For Player {}:\n{}\n",
                             other_player,
                             state.ObservationString(other_player));
            }
          }
        }
      }
      if (print_tensor_repr) {
        if (game_type.provides_information_state_tensor) {
          it = format_to(it, "Info State Tensor For Current Player:\n{}\n",
                         state.InformationStateTensor(player));
          if (show_all_perspectives) {
            for (auto other_player = open_spiel::Player{0};
                 other_player < game->NumPlayers(); other_player++) {
              if (other_player == player)
                continue;
              it = format_to(it, "Info State Tensor For Player {}:\n{}\n",
                             other_player,
                             state.InformationStateTensor(other_player));
            }
          }
        }
        if (game_type.provides_observation_tensor) {
          it = format_to(it, "Observation Tensor For Current Player:\n{}\n",
                         state.ObservationTensor(player));
          if (show_all_perspectives) {
            for (auto other_player = open_spiel::Player{0};
                 other_player < game->NumPlayers(); other_player++) {
              if (other_player == player)
                continue;
              it = format_to(it, "Observation Tensor For Player {}:\n{}\n",
                             other_player,
                             state.ObservationTensor(other_player));
            }
          }
        }
      }

      if (print_legal_actions) {
        it = format_to(it, "Legal Actions:\n");
        auto actions = state.LegalActions(player);
        auto action_strs_rng = actions | ranges::views::transform([&state](open_spiel::Action act) {
              return state.ActionToString(state.CurrentPlayer(), act);
            });
        it = format_to(it, "{}\n", fmt::join(action_strs_rng, ", "));
      }

    }
    return it;
  }
};

} // namespace fmt
