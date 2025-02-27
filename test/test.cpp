#include "test.h"

// Function definitions
int add(int i, int j) {
    return i + j;
}

torch::Tensor minimal_create_tensor(int dim) {
    return torch::randn({3, 3});
}

torch::Tensor get_arg_max(torch::Tensor x) {
    return torch::argmax(x, 1);
}

void learn(ReplayBuffer &rp, Policy pol, Policy critic, size_t num_it) {
    auto ds = LLDerived(rp);
    size_t batch_size{4};
    std::cout << ds.size().value() << std::endl;

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(ds),
        batch_size
    );

    for (auto &batch : *data_loader) {
        auto d = batch.data();
        std::cout << "in batch: \n";
        std::cout << std::get<0>(*d) << std::endl;
        std::cout << std::get<1>(*d) << std::endl;
        std::cout << std::get<2>(*d) << std::endl;
        std::cout << std::get<3>(*d) << std::endl;
    }
}

// Class definitions
MDPTransition::MDPTransition(Eigen::VectorXd s, int a, double r, Eigen::VectorXd s_prime)
    : state(std::move(s)), action(a), reward(r), next_state(std::move(s_prime)) {}

MDPTransition::MDPTransition(py::array_t<double> s, int a, double r, py::array_t<double> s_prime) {
    state = Eigen::Map<const Eigen::VectorXd>(s.data(), s.size()).eval();
    next_state = Eigen::Map<const Eigen::VectorXd>(s_prime.data(), s_prime.size()).eval();
    action = a;
    reward = r;
}

MDPTransition::MDPTransition(const MDPTransition &other)
    : state(other.state), action(other.action), reward(other.reward), next_state(other.next_state) {}

MDPTransition::MDPTransition(MDPTransition &&other) noexcept
    : state(std::move(other.state)),
      action(std::exchange(other.action, 0)),
      reward(std::exchange(other.reward, 0)),
      next_state(std::move(other.next_state)) {}

MDPTransition& MDPTransition::operator=(MDPTransition &&other) noexcept {
    if (this != &other) {
        state = std::move(other.state);
        action = std::exchange(other.action, 0);
        reward = std::exchange(other.reward, 0.0);
        next_state = std::move(other.next_state);
    }
    return *this;
}

std::tuple<Eigen::VectorXd, int, double, Eigen::VectorXd> MDPTransition::get() {
    return std::make_tuple(state, action, reward, next_state);
}

Eigen::VectorXd MDPTransition::get_state() const { return state; }
Eigen::VectorXd MDPTransition::get_next_state() const { return next_state; }
int MDPTransition::get_action() const { return action; }
double MDPTransition::get_reward() const { return reward; }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MDPTransition::to_tens() {
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor state_tens = torch::from_blob(state.data(), {state.rows(), state.cols()}, options).to(torch::kFloat32);
    torch::Tensor nstate_tens = torch::from_blob(next_state.data(), {next_state.rows(), next_state.cols()}, options).to(torch::kFloat32);
    torch::Tensor action_tens = torch::tensor({action});
    torch::Tensor reward_tens = torch::tensor({reward});
    return std::make_tuple(state_tens, action_tens, reward_tens, nstate_tens);
}

ReplayBuffer::ReplayBuffer(size_t len) : hist_size(len) {
    buf.reserve(len);
    len = 0;
}

ReplayBuffer::ReplayBuffer() : hist_size(100000) {
    buf.reserve(hist_size);
    len = 0;
}

void ReplayBuffer::append_to_buffer(MDPTransition &trans) {
    buf.push_back(std::move(trans));
    len += buf.size();
}

std::vector<MDPTransition> ReplayBuffer::get() {
    return buf;
}

int ReplayBuffer::get_length() {
    return len;
}

LLDerived::LLDerived(ReplayBuffer &rp) : replay(rp) {
    size_of_data = rp.get_length();
}

CustExample LLDerived::get(size_t index)  {
    return replay.get()[index].to_tens();
}

size_t LLDerived::len() {
    return size_of_data;
}

std::optional<size_t> LLDerived::size() const  {
    return size_of_data;
}

Policy::Policy()
    : linear_1(register_module("linear_1", torch::nn::Linear(d_inp, pow(2, 8)))),
      linear_2(register_module("linear_2", torch::nn::Linear(pow(2, 8), pow(2, 6)))),
      linear_3(register_module("linear_3", torch::nn::Linear(pow(2, 6), d_action))) {}

torch::Tensor Policy::forward(torch::Tensor x) {
    x = linear_1(x);
    x = linear_2(x);
    x = linear_3(x);
    return x;
}
