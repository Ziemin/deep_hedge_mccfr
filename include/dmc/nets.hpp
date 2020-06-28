#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <torch/all.h>

namespace dmc::nets {

struct StackedLinearNet : torch::nn::Module {
  using FeaturesType = torch::Tensor;

  StackedLinearNet(uint32_t features_size, uint32_t output_size,
                   torch::ArrayRef<uint32_t> hidden_units,
                   bool use_skip_connections)
      : fc_out(torch::nn::LinearOptions(*hidden_units.rbegin(), output_size)
               .bias(true)),
        use_skip_connections(use_skip_connections)
  {

    uint32_t prev_units = features_size;
    uint32_t layer_num = 1;
    for (uint32_t out_units : hidden_units) {
      torch::nn::Linear fc(
          torch::nn::LinearOptions(prev_units, out_units).bias(true));
      register_module(fmt::format("fc_{}", layer_num), fc);
      hidden_layers.push_back(fc);
      torch::nn::init::xavier_normal_(fc->weight);

      prev_units = out_units;
      layer_num += 1;
    }
    register_module("fc_out", fc_out);
    torch::nn::init::xavier_normal_(fc_out->weight);
  }

  torch::Tensor forward(torch::Tensor features) {
    using namespace torch::indexing;
    auto out = features;
    for (auto &layer : hidden_layers) {
      auto next_out = torch::relu(layer(out));
      if (use_skip_connections) {
        auto pad = next_out.size(1) - out.size(1);
        if (pad > 0) {
          out = torch::constant_pad_nd(out, {0, pad, 0, 0}, 0);
        } else if(pad < 0) {
          out = out.index({"...", Slice(0, next_out.size(1))});
        }
        out = next_out + out;
      } else {
        out = next_out;
      }
    }
    return fc_out(out);
  }

private:
  std::vector<torch::nn::Linear> hidden_layers;
  torch::nn::Linear fc_out;
  bool use_skip_connections;
};

struct HoldemNet : torch::nn::Module {

  using FeaturesType = torch::Tensor;

  HoldemNet(uint32_t num_players, uint32_t cards_size, uint32_t actions_size,
            uint32_t output_size, uint32_t embed_num_layers,
            uint32_t joint_num_layers, uint32_t card_dim, uint32_t action_dim,
            uint32_t joint_dim, bool normalize)
      : num_players(num_players), cards_size(cards_size),
        actions_size(actions_size),
        out_head(torch::nn::Linear(torch::nn::LinearOptions(joint_dim, output_size).bias(true))),
        normalize(normalize) {
    if (joint_num_layers < 1) {
      throw std::invalid_argument("joint_num_layers has to be greater or equal than 1");
    }
    if (embed_num_layers < 1) {
      throw std::invalid_argument(
          "embed_num_layers has to be greater or equal than 1");
    }
    for (uint32_t ind = 0; ind < embed_num_layers; ind++) {
      uint32_t card_in_dim = card_dim;
      uint32_t action_in_dim = action_dim;
      if (ind == 0) {
        card_in_dim = cards_size;
        action_in_dim = actions_size;
      }
      // create card embedding layer
      torch::nn::Linear fcc(
          torch::nn::LinearOptions(cards_size, card_dim).bias(true));
      register_module(fmt::format("card_fc_{}", ind), fcc);
      card_layers.push_back(fcc);

      // create actions embedding layer
      torch::nn::Linear fca(
          torch::nn::LinearOptions(actions_size, action_dim).bias(true));
      register_module(fmt::format("action_fc_{}", ind), fca);
      action_layers.push_back(fca);
    }
    for (uint32_t ind = 0; ind < joint_num_layers; ind++) {
      uint32_t joint_in_dim = joint_dim;
      if (ind == 0) {
        joint_in_dim = card_dim + action_dim;
      }
      torch::nn::Linear fc(
          torch::nn::LinearOptions(joint_in_dim, joint_dim).bias(true));
      register_module(fmt::format("joint_fc_{}", ind), fc);
      joint_layers.push_back(fc);
    }

    register_module("output_head", out_head);
  }

  torch::Tensor forward(torch::Tensor input) {

    using namespace torch::indexing;
    auto card_data = input.index({"...", Slice(num_players, num_players + cards_size)});
    auto action_data =
        input.index({"...", Slice(num_players + cards_size, None)});

    // prepare card features
    for (auto &layer : card_layers) {
      card_data = torch::relu(layer(card_data));
    }
    // prepare action features
    for (auto &layer : action_layers) {
      action_data = torch::relu(layer(action_data));
    }

    // run joint features layers
    auto joint_features = torch::cat({card_data, action_data}, 1);
    joint_features = torch::relu(joint_layers[0](joint_features));

    for (size_t ind = 1; ind < joint_layers.size(); ind++) {
      // skip connections
      joint_features =
          torch::relu(joint_layers[ind](joint_features) + joint_features);
    }
    if (normalize) {
      joint_features = (joint_features - joint_features.mean({1}, true)) /
                       (joint_features.std({1}, true, true) + 1e-7);
    }
    // prepare network output
    return out_head(joint_features);
  }

private:
  uint32_t num_players, cards_size, actions_size;
  std::vector<torch::nn::Linear> card_layers;
  std::vector<torch::nn::Linear> action_layers;
  std::vector<torch::nn::Linear> joint_layers;
  torch::nn::Linear out_head;
  bool normalize;
};

} // namespace dmc::nets
