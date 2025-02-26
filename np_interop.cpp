#include <ATen/ops/argmax.h>
#include <ATen/ops/linear.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/linear.h>
#include<array>
#include <torch/torch.h>
#include <torch/extension.h>
#include <Eigen/Dense>
#include <Eigen/Core>
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
};



class ReplayBuffer {
private:
std::vector<MDPTransition> buf;
const size_t hist_size;

public:
    ReplayBuffer(size_t len) : hist_size (len) {
        buf.reserve(len);
    };

    ReplayBuffer() :hist_size(100000){
        // buf = std::vector<MDPTransition>();
        buf.reserve(hist_size);
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
};

// realizable goal: learn with N agents at once, syncying gradients after a number of steps 
class Policy: public torch::nn::Module {
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
    Policy()
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
        //const torch::nn::functional::SoftmaxFuncOptions so(1);
        //auto probs = torch::nn::functional::softmax(x, so);
        return x;
        // return torch::argmax(probs, 1);
    };
};


torch::Tensor minimal_create_tensor(int dim){

    return torch::randn({3,3});
}


torch::Tensor get_arg_max(torch::Tensor x){
    return torch::argmax(x, 1);

};


void learn(ReplayBuffer &rp, Policy pol){
    std::cout << "Training."
    ;
};


PYBIND11_MODULE(np_interop, m){
    // first arg: module name, not in quotes
    // second arg: define variable of type py::module_ <- interface for creating bindings
    m.doc() = "pybind11 sample module";
    m.def("add", &add, "A function adding two integers i, j.");
    m.def("minimal_tensor_create", &minimal_create_tensor, "Create minimal tensor");
    m.def("get_arg_max", &get_arg_max, "get argmax of tensor");
    m.def("train", &learn, "Train policy on replay buffer.");

    pybind11::class_<Policy, std::shared_ptr<Policy>, torch::nn::Module>(m, "Policy")
        .def(py::init())
        .def("forward", &Policy::forward);

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
        .def("get_reward", &MDPTransition::get_reward);
}
