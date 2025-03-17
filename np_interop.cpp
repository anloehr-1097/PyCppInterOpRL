#include <ATen/Functions.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/stack.h>
#include <c10/core/Scalar.h>
#include <cassert>
#include <memory>
#include <ostream>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <torch/data.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader/base.h>
#include <torch/data/dataloader/stateful.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include<array>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/types.h>
#include <tuple>
#include <utility>
#include <vector>

class MDPTransition {
    /* 
     * Representing single MDPTransition
     * */
public:
    Eigen::VectorXd state; 
    Eigen::VectorXd next_state; 
    int action;
    double reward;

    MDPTransition(Eigen::VectorXd s, int a, double r, Eigen::VectorXd s_prime) {
        // empyt init, assigning to const members
        state = std::move(s);
        action = a;
        reward = r;
        next_state = std::move(s_prime);
    };

    MDPTransition(py::array_t<double> s, int a, double r, py::array_t<double>s_prime) {
        state = Eigen::Map<const Eigen::VectorXd>(s.data(), s.size()).eval();
        next_state = Eigen::Map<const Eigen::VectorXd>(s_prime.data(), s_prime.size()).eval();
        action = a;
        reward = r;
    };

    // copy constructor
    MDPTransition(const MDPTransition &other)
       : state(other.state), action(other.action), reward(other.reward), next_state(other.next_state){}

    // move constructor
    MDPTransition(MDPTransition &&other) noexcept
        : state(std::move(other.state)),
        action(std::exchange(other.action, 0)),
        reward(std::exchange(other.reward, 0)),
        next_state(std::move(other.next_state))
    {};

    // move assignment operator
    MDPTransition& operator=(MDPTransition &&other) noexcept {
        if (this != &other){
        state = std::move(other.state);
        action = std::exchange(other.action, 0);
        reward = std::exchange(other.reward, 0.0);
        next_state = std::move(other.next_state);
        }
        return *this;
    };

    std::tuple<Eigen::VectorXd, int ,double, Eigen::VectorXd> get(){
        return std::make_tuple(state, action, reward, next_state);
    };

    Eigen::VectorXd get_state() const { return state; }
    Eigen::VectorXd get_next_state() const { return next_state; }
    int get_action() const { return action; }
    double get_reward() const { return reward; } 

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> to_tens(){
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor state_tens = torch::from_blob(state.data(), {state.rows(), state.cols()}, options).to(torch::kFloat32);

        torch::Tensor nstate_tens = torch::from_blob(next_state.data(), {next_state.rows(), next_state.cols()}, options).to(torch::kFloat32);
        torch::Tensor action_tens = torch::tensor({action});
        torch::Tensor reward_tens = torch::tensor({reward}); 
        return std::make_tuple(state_tens, action_tens, reward_tens, nstate_tens);
    };
};



class ReplayBuffer {
    /*
     * ReplayBuffer to capture trajectories.
     */
private:
std::vector<MDPTransition> buf;
const size_t hist_size;
size_t len = 0;

public:
    ReplayBuffer(size_t len) : hist_size (len) {
        buf.reserve(len);
        len = 0;
    };

    ReplayBuffer() :hist_size(100000){
        buf.reserve(hist_size);
        len = 0;
    };

    void append_to_buffer(MDPTransition &trans){
        // pass a reference and move the object to vector instead of copying it
        buf.push_back(std::move(trans));
        return;
    };

    std::vector<MDPTransition> get() {
        return buf;
    };

    size_t get_length(){
        return buf.size();
    };

    void clear(){
        buf.clear();
    };
};


// custom Example used in custom dataset class
using CustExample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

// custom dataset class used for RL task
class LLDerived: public torch::data::Dataset<LLDerived, CustExample> {
private:
    ReplayBuffer replay;
    size_t size_of_data;
public:

    explicit LLDerived(ReplayBuffer &rp) : 
        replay(rp),
        size_of_data(rp.get_length())
    {
        std::cout << "Dataset Length: " << size_of_data << std::endl;
    };

    CustExample get(size_t index) override{
        // return data as tensors
        auto mem = replay.get()[index].to_tens();
        return mem;
    };

    size_t len() {
        return size_of_data;
    }

    std::optional<size_t> size() const override {
        return size_of_data;
    };
};


// realizable goal: learn with N agents at once, syncying gradients after a number of steps 
class PolicyImpl: public torch::nn::Module {
    /*
     * Policy network param for LunarLander (with fixed number of 4 actions)
     * */
    int d_inp {8};
    int d_out {4};
    const int d_action {4};
    torch::nn::Linear linear_1;
    torch::nn::Linear linear_2;
    torch::nn::Linear linear_3;
    std::array<int, 4> action_space {1, 2, 3, 4};
    
public:
    PolicyImpl() : 
        linear_1(register_module("linear_1", torch::nn::Linear(d_inp, pow(2,8)))),
        linear_2(register_module("linear_2", torch::nn::Linear(pow(2,8), pow(2,6)))),
        linear_3(register_module("linear_3", torch::nn::Linear(pow(2,6), d_action)))
    {
        ;
    };

    torch::Tensor forward(torch::Tensor x){
        x = linear_1(x);
        x = linear_2(x);
        x = linear_3(x);
        return x;
        // softmax not applied, done later
    };
};


class CustColl: public torch::data::transforms::Collation<CustExample> {
    /* Collator for cust data batches.*/
public:
    ~CustColl(){
        ;
    };

    CustExample apply_batch(std::vector<CustExample> cust_example_v)
    {
        std::vector<torch::Tensor> states, actions, rewards, next_states;
        for (CustExample ex: cust_example_v){
            auto &&[state, action, reward, next_state] = ex;
            states.push_back(state);
            actions.push_back(action);
            rewards.push_back(reward);
            next_states.push_back(next_state);
        };

        torch::Tensor state_t = torch::squeeze(torch::stack(states), -1);
        torch::Tensor action_t = torch::squeeze(torch::stack(actions), -1);
        torch::Tensor reward_t = torch::squeeze(torch::stack(rewards), -1);
        torch::Tensor next_state_t = torch::squeeze(torch::stack(next_states), -1);

        return std::make_tuple(
            state_t,
            action_t,
            reward_t,
            next_state_t
        );
    };
};

torch::Tensor get_arg_max(torch::Tensor x){
    return torch::argmax(x, 1);
};

torch::Tensor q_learning_loss(
    torch::Tensor immediate_return,
    torch::Tensor q_values,
    torch::Tensor target_q_values,
    float discount_factor)
{
    torch::Tensor target_state_action_return = immediate_return + discount_factor * torch::argmax(target_q_values, 1);
    return torch::nn::functional::mse_loss(q_values, target_state_action_return).to(torch::kF32);
};

void learn(ReplayBuffer &rp, PolicyImpl pol, PolicyImpl critic, size_t num_it, size_t batch_size){
    // learn with replay buffer. Callable from python.
    py::gil_scoped_release release;
    auto ds = LLDerived(rp).map(CustColl());

    torch::optim::SGDOptions optim_opts(0.0001  /* learning rate */);
    torch::optim::SGD optim(pol.parameters(), optim_opts);
    optim.zero_grad();

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(ds), 
        batch_size
    );
    pol.train();
    critic.eval();
    torch::Tensor total_loss = torch::tensor({0.0}, torch::kFloat32); 
    int count = 0;
    for (CustExample &batch: *data_loader){
        // batch = vector(Tuples)
        optim.zero_grad();
        count += 1;
        const auto[cur_obs, action, reward, next_obs] = batch;  // unpack
        auto range = torch::arange(action.size(0));
        
        // calc preds, loss
        torch::Tensor q_values = pol.forward(cur_obs).index({range, action});
        torch::Tensor target_q_values = critic.forward(next_obs);
        torch::Tensor loss = q_learning_loss(reward, q_values, target_q_values, 0.95);

        if (torch::isnan(loss).item<bool>()) {
            // std::cout << "Nan detected";
            continue;
        }

        std::cout << "Loss: " << loss;
        loss.backward();
        // grad clipping
        torch::nn::utils::clip_grad_norm_(pol.parameters(), 1.0);
        optim.step();
        optim.zero_grad();
        total_loss += loss.detach();
    };
};


void transfer_state_dict(std::shared_ptr<PolicyImpl> source, std::shared_ptr<PolicyImpl> dest){
    // transfer source to dest model, callable from python
    std::string tmp_name {"temp_model.pt"};
    torch::save(source, tmp_name);
    torch::load(dest, tmp_name);
};

void load_from_file(std::shared_ptr<PolicyImpl> model, std::string fname){
    // load model from file, callable from python
    torch::load(model, fname);
}

void save_to_file(std::shared_ptr<PolicyImpl> model, std::string fname){
    // save model to file, callable from python
    torch::save(model, fname);
}


PYBIND11_MODULE(np_interop, m){
    // first arg: module name, not in quotes
    // second arg: define variable of type py::module_ <- interface for creating bindings
    m.doc() = "pybind11 sample module";
    m.def("get_arg_max", &get_arg_max, "get argmax of tensor");
    m.def("train", &learn, "Train policy on replay buffer.");
    m.def("transfer_state_dict", &transfer_state_dict, "Transfer state dict from src to dest model.");
    m.def("load_from_file", &load_from_file, "Load from file into model.");
    m.def("save_to_file", &save_to_file, "Save model in file.");

    pybind11::class_<PolicyImpl, std::shared_ptr<PolicyImpl>, torch::nn::Module>(m, "Policy")
        .def(py::init())
        .def("forward", &PolicyImpl::forward);

    pybind11::class_<ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<size_t>())
        .def(py::init())
        .def("append", &ReplayBuffer::append_to_buffer, "append to buffer")
        .def("as_list", &ReplayBuffer::get, "Get replay buffer")
        .def("get_length", &ReplayBuffer::get_length, "Get replay buffer length")
        .def("clear", &ReplayBuffer::clear, "Clear replay buffer");

    pybind11::class_<MDPTransition>(m, "MDPTransition")
        .def(py::init<py::array_t<double>, int, double, py::array_t<double>>())
        .def("get", &MDPTransition::get)
        .def("get_state", &MDPTransition::get_state)
        .def("get_next_state", &MDPTransition::get_next_state)
        .def("get_action", &MDPTransition::get_action)
        .def("get_reward", &MDPTransition::get_reward)
        .def("to_tens", &MDPTransition::to_tens);
}
