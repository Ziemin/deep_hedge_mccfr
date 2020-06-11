#pragma once
#include <torch/torch.h>
#include <vector>

namespace dmc::data {

template <typename ElementType>
class VectorDataset
    : public torch::data::datasets::Dataset<VectorDataset<ElementType>, ElementType> {
private:
  std::vector<ElementType> data_;

public:
  VectorDataset(std::vector<ElementType> data) : data_(std::move(data)) {}

  ElementType get(size_t index) override { return data_[index]; }

  torch::optional<size_t> size() const override { return data_.size(); }
};

} // namespace dmc::data
