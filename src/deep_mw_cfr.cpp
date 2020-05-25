#include <deep_mw_cfr.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>

torch::Tensor some_fun() {
  return torch::eye(10);
}
