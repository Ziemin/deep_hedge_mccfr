#include <dhc/deep_hedge_mccfr.hpp>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace dhc {

using namespace nlohmann;

json SolverSpec::to_json() const {
  return {{"max_steps", max_steps},
          {"player_traversals", player_traversals},
          {"baseline_traversals", baseline_traversals},
          {"baseline_start_step", baseline_start_step},
          {"player_update_freq", player_update_freq},

          {"player_lr_init", player_lr_init},
          {"baseline_lr_init", baseline_lr_init},
          {"player_lr_end", player_lr_end},
          {"baseline_lr_end", baseline_lr_end},
          {"decay_steps", decay_steps},
          {"decay_rate", decay_rate},
          {"gradient_clipping_value", gradient_clipping_value},
          {"logits_threshold", logits_threshold},
          {"weight_decay", weight_decay},
          {"entropy_cost", entropy_cost},

          {"eta", eta},
          {"normalize_returns", normalize_returns},

          {"epsilon", epsilon},
          {"seed", seed}};
}

SolverSpec SolverSpec::from_json(const json &spec_json) {
  SolverSpec spec;

  auto maybe_assign =
    [&spec_json](std::string_view name, auto& field) {
      if (auto it = spec_json.find(name); it != spec_json.end()) {
        field = it->get<typename std::remove_reference<decltype(field)>::type>();
      }
    };

  maybe_assign("max_steps", spec.max_steps);
  maybe_assign("player_traversals", spec.player_traversals);
  maybe_assign("baseline_traversals", spec.baseline_traversals);
  maybe_assign("baseline_start_step", spec.baseline_start_step);
  maybe_assign("player_update_freq", spec.player_update_freq);

  maybe_assign("player_lr_init", spec.player_lr_init);
  maybe_assign("baseline_lr_init", spec.baseline_lr_init);
  maybe_assign("player_lr_end", spec.player_lr_end);
  maybe_assign("baseline_lr_end", spec.baseline_lr_end);
  maybe_assign("decay_steps", spec.decay_steps);
  maybe_assign("decay_rate", spec.decay_rate);
  maybe_assign("gradient_clipping_value", spec.gradient_clipping_value);
  maybe_assign("logits_threshold", spec.logits_threshold);
  maybe_assign("weight_decay", spec.weight_decay);
  maybe_assign("entropy_cost", spec.entropy_cost);

  maybe_assign("eta", spec.eta);
  maybe_assign("normalize_returns", spec.normalize_returns);
  maybe_assign("epsilon", spec.epsilon);
  maybe_assign("seed", spec.seed);

  return spec;
}

} // namespace dhc
