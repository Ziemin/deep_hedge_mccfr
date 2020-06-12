#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <torch/all.h>

namespace dmc::nets {

struct StackedLinearNet : torch::nn::Module {
  using FeaturesType = torch::Tensor;

  StackedLinearNet(uint32_t features_size, uint32_t output_size,
                   torch::ArrayRef<uint32_t> hidden_units)
      : fc_out(torch::nn::LinearOptions(*hidden_units.rbegin(), output_size)
                   .bias(true)) {

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
    auto out = features;
    for (auto &layer : hidden_layers) {
      out = torch::relu(layer(out));
    }
    return fc_out(out);
  }

private:
  std::vector<torch::nn::Linear> hidden_layers;
  torch::nn::Linear fc_out;
};

} // namespace dmc::nets
