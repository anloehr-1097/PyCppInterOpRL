#ifndef TEST_H
#define TEST_H

#include <ATen/Functions.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/linear.h>
#include <c10/core/Scalar.h>
#include <torch/data.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader/base.h>
#include <torch/data/dataloader/stateful.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include <array>
#include <torch/torch.h>
#include <torch/extension.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/types.h>
#include <tuple>
#include <vector>

// Function declarations
int add(int i, int j);
torch::Tensor minimal_create_tensor(int dim);
torch::Tensor get_arg_max(torch::Tensor x);

// Class declarations
class MDPTransition {
public:
    Eigen::VectorXd state;
    Eigen::VectorXd next_state;
    int action;
    double reward;

    MDPTransition(Eigen::VectorXd s, int a, double r, Eigen::VectorXd s_prime);
    MDPTransition(py::array_t<double> s, int a, double r, py::array_t<double> s_prime);
    MDPTransition(const MDPTransition &other);
    MDPTransition(MDPTransition &&other) noexcept;
    MDPTransition& operator=(MDPTransition &&other) noexcept;
    std::tuple<Eigen::VectorXd, int, double, Eigen::VectorXd> get();
    Eigen::VectorXd get_state() const;
    Eigen::VectorXd get_next_state() const;
    int get_action() const;
    double get_reward() const;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> to_tens();
};

class ReplayBuffer {
private:
    std::vector<MDPTransition> buf;
    const size_t hist_size;
    size_t len;

public:
    ReplayBuffer(size_t len);
    ReplayBuffer();
    void append_to_buffer(MDPTransition &trans);
    std::vector<MDPTransition> get();
    int get_length();
};

using CustExample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
class LLDerived : public torch::data::Dataset<LLDerived, CustExample> {
private:
    ReplayBuffer replay;
    int size_of_data;

public:
    explicit LLDerived(ReplayBuffer &rp);
    CustExample get(size_t index) override;
    size_t len();
    std::optional<size_t> size() const override;
};

class Policy : public torch::nn::Module {
    int d_inp{8};
    int d_out{4};
    const int d_action{4};
    torch::nn::Linear linear_1;
    torch::nn::Linear linear_2;
    torch::nn::Linear linear_3;
    std::array<int, 4> action_space{1, 2, 3, 4};

public:
    Policy();
    torch::Tensor forward(torch::Tensor x);
};

void learn(ReplayBuffer &rp, Policy pol, Policy critic, size_t num_it);
#endif // TEST_H
