#include <dmc/deep_mw_cfr.hpp>
#include <open_spiel/spiel.h>
#include <range/v3/all.hpp>
#include <torch/torch.h>

namespace dmc {

torch::Tensor some_fun() { return torch::eye(10); }

} // namespace dmc
