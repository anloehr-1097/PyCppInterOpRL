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

int add(int i, int j){
    return i+j;
}

class MDPTransition {

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
        // buf = std::vector<MDPTransition>();
        buf.reserve(hist_size);
        len = 0;
    };

    void append_to_buffer(MDPTransition &trans){
        // pass a reference and move the object to vector instead of copying it
        buf.push_back(std::move(trans));
        // buf.insert(buf.end(), trans);
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


using CustExample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
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
    // torch::data::Example<> get(size_t index) override;
    // CustExample get_batch(size_t index) {
        // return data as tensors
        auto mem = replay.get()[index].to_tens();
    //     // torch::Tensor t0, t1, t2, t3;
    //     // t0 = std::get<0>(mem);
    //     // t1 = std::get<1>(mem);
    //     // t2 = std::get<2>(mem);
    //     // t3 = std::get<3>(mem);
    //     // return {t0, t1, t2, t3};
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
    int d_inp {8};
    int d_out {4};
    const int d_action {4};
    torch::nn::Linear linear_1;
    torch::nn::Linear linear_2;
    torch::nn::Linear linear_3;
    // torch::nn::Softmax final_softmax;
    // mapping from state to action
    // potentially using  
    std::array<int, 4> action_space {1, 2, 3, 4};
    
public:
    PolicyImpl()
    : linear_1(register_module("linear_1", torch::nn::Linear(d_inp, pow(2,8)))),
        linear_2(register_module("linear_2", torch::nn::Linear(pow(2,8), pow(2,6)))),
        linear_3(register_module("linear_3", torch::nn::Linear(pow(2,6), d_action)))
        // final_softmax(register_module("final_softmax", torch::nn::Softmax(0)))
    {
        // initialize a policy
        // this policy is a simple neural net producing a single integer output
    };

    torch::Tensor forward(torch::Tensor x){
        x = linear_1(x);
        x = linear_2(x);
        x = linear_3(x);
        // auto probs = final_softmax(x);
        //const torch::nn::functional::SoftmaxFuncOptions so(1); auto probs = torch::nn::functional::softmax(x, so); return x;
        // return torch::argmax(probs, 1);
        return x;
    };

    void save(std::string file_path){
        torch::serialize::OutputArchive output_archive;
        this->save(file_path);
    }

};

// TORCH_MODULE(Policy);




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

        std::cout << "state_t size: " << state_t.sizes() << std::endl;
        std::cout << "action_t size: " << action_t.sizes() << std::endl;
        std::cout << "reward_t size: " << reward_t.sizes() << std::endl;
        std::cout << "next_state_t size: " << next_state_t.sizes() << std::endl;

        return std::make_tuple(
            state_t,
            action_t,
            reward_t,
            next_state_t
        );
    };
};


torch::Tensor minimal_create_tensor(int dim){

    return torch::randn({3,3});
}


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
    // std::cout << "q_learning_loss: ";
    // std::cout << q_values.sizes() << " ,";
    // std::cout << target_q_values.sizes() << " ,";
    // std::cout << immediate_return.sizes() << " ,";
    // std::cout << torch::argmax(target_q_values, 1).sizes() << std::endl;
    // std::cout << target_state_action_return.sizes() << std::endl;
    return torch::nn::functional::mse_loss(q_values, target_state_action_return).to(torch::kF32);
};


void learn(ReplayBuffer &rp, PolicyImpl pol, PolicyImpl critic, size_t num_it, size_t batch_size){

    py::gil_scoped_release release;
    auto ds = LLDerived(rp).map(CustColl());
    std::cout << ds.size().value() << std::endl;

    torch::optim::SGDOptions optim_opts(0.001  /* learning rate */);
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
        std::cout << "Batch number: " << count << " with batch size " << std::get<0>(batch).size(0) <<".\n";
        count += 1;
        const auto[cur_obs, action, reward, next_obs] = batch;
        auto range = torch::arange(action.size(0));
        

        //torch::indexing::get_item(policy.forward(cur_obs), );
        torch::Tensor q_values = pol.forward(cur_obs).index({range, action});
        // TODO: implement the following, q_values passed are not correct here
        // q_values = policy(cur_obs)[torch.arange(action.shape[0]), action].to(torch.float)
        // torch::Tensor q_values = pol.forward(cur_obs);
        torch::Tensor target_q_values = critic.forward(next_obs);
        // std::cout << "q_values.sizes() = " << q_values.sizes() << std::endl;
        // std::cout << "target_q_values.sizes() = " << target_q_values.sizes() << std::endl;
        // std::cout << "reward.sizes()" << reward.sizes() << std::endl;
        torch::Tensor loss = q_learning_loss(reward, q_values, target_q_values, 0.95);
        std::cout << "Loss: " << loss;

        loss.backward();
        optim.step();
        optim.zero_grad();
        total_loss += loss;
    };
};


void transfer_state_dict(std::shared_ptr<PolicyImpl> source, std::shared_ptr<PolicyImpl> dest){
    std::string tmp_name {"temp_model.pt"};
    source->save(tmp_name);
    torch::load(dest, tmp_name);
    // torch::load(dest, tmp_name);
};


PYBIND11_MODULE(np_interop, m){
    // first arg: module name, not in quotes
    // second arg: define variable of type py::module_ <- interface for creating bindings
    m.doc() = "pybind11 sample module";
    m.def("add", &add, "A function adding two integers i, j.");
    m.def("minimal_tensor_create", &minimal_create_tensor, "Create minimal tensor");
    m.def("get_arg_max", &get_arg_max, "get argmax of tensor");
    m.def("train", &learn, "Train policy on replay buffer.");
    m.def("transfer_state_dict", &transfer_state_dict, "Transfer state dict from src to dest model.");

    pybind11::class_<PolicyImpl, std::shared_ptr<PolicyImpl>, torch::nn::Module>(m, "Policy")
        .def(py::init())
        .def("forward", &PolicyImpl::forward);

    // pybind11::class_<LLDerived, std::shared_ptr<LLDerived>>(m, "LunarLanderData")
    //     .def(py::init<ReplayBuffer &, int>());
    //     //.def("forward", &Policy::forward);

    pybind11::class_<ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<size_t>())
        .def(py::init())
        .def("append", &ReplayBuffer::append_to_buffer, "append to buffer")
        .def("as_list", &ReplayBuffer::get, "Get replay buffer");

    pybind11::class_<MDPTransition>(m, "MDPTransition")
        // .def(py::init<Eigen::VectorXd, int, double>())
        .def(py::init<py::array_t<double>, int, double, py::array_t<double>>())
        .def("get", &MDPTransition::get)
        .def("get_state", &MDPTransition::get_state)
        .def("get_next_state", &MDPTransition::get_next_state)
        .def("get_action", &MDPTransition::get_action)
        .def("get_reward", &MDPTransition::get_reward)
        .def("to_tens", &MDPTransition::to_tens);
}
