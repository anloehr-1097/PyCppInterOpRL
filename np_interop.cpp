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
#include <vector>

int add(int i, int j){
    return i+j;
}


class MDPTransition {

public:
    const Eigen::VectorXd state; 
    const int action;
    const double reward;


    MDPTransition(Eigen::VectorXd s, int a, double r) :
        state(s),
        action(a),
        reward(r)
    {
        // empyt init, assigning to const members
        ;
    };
    std::tuple<Eigen::VectorXd, int ,double> get(){

        return std::make_tuple(state, action, reward);

    };
};



class ReplayBuffer {
std::vector<std::vector<double>> buf;
const size_t hist_size = 10000;

public:
    ReplayBuffer(size_t len) : hist_size (len) {
        auto buf = std::vector<std::vector<double>>(len);
    };

    ReplayBuffer() {
        auto buf = std::vector<std::vector<double>>();
    };

    void append_to_buffer(Eigen::VectorXd &vec){
        const std::vector<double> res(vec.data(), vec.data() + vec.size());
        buf.insert(buf.end(), res);
        return;
    };

    Eigen::MatrixXd to_numpy(){
        // convert buffer to np array
        const size_t cols = buf.at(0).size();
        const size_t rows = buf.size();
        Eigen::MatrixXd a(rows, cols);

        for (size_t i = 0; i < rows; ++i){
            // v is vector
            a.row(i) = Eigen::Map<Eigen::VectorXd>(buf[i].data(), cols);
        }
        return a;
    };
};

// realizable goal: learn with N agents at once, syncying gradients after a number of steps 
class Policy: public torch::nn::Module {
    int d_inp {8};
    int d_out {1};
    const int d_action {4};
    torch::nn::Linear linear_1;
    torch::nn::Linear linear_2;
    torch::nn::Linear linear_3;
    torch::nn::Softmax final_softmax;
    // mapping from state to action
    // potentially using  
    std::array<int, 4> action_space {1, 2, 3, 4};
    
public:
    Policy()
    : linear_1(register_module("linear_1", torch::nn::Linear(d_inp, pow(2,8)))),
        linear_2(register_module("linear_2", torch::nn::Linear(pow(2,8), pow(2,6)))),
        linear_3(register_module("linear_3", torch::nn::Linear(pow(2,6), d_action))),
        final_softmax(register_module("final_softmax", torch::nn::Softmax(0)))
    {
        // initialize a policy
        // this policy is a simple neural net producing a single integer output
    };

    torch::Tensor forward(torch::Tensor x){
        x = linear_1(x);
        x = linear_2(x);
        x = linear_3(x);
        // auto probs = final_softmax(x);
        const torch::nn::functional::SoftmaxFuncOptions so(1);
        auto probs = torch::nn::functional::softmax(x, so);
        return probs;
        // return torch::argmax(probs, 1);
    };
};


torch::Tensor minimal_create_tensor(int dim){

    return torch::randn({3,3});
}


torch::Tensor get_arg_max(torch::Tensor x){
    return torch::argmax(x, 1);

};


PYBIND11_MODULE(np_interop, m){
    // first arg: module name, not in quotes
    // second arg: define variable of type py::module_ <- interface for creating bindings
    m.doc() = "pybind11 sample module";
    m.def("add", &add, "A function adding two integers i, j.");
    m.def("minimal_tensor_create", &minimal_create_tensor, "Create minimal tensor");
    m.def("get_arg_max", &get_arg_max, "get argmax of tensor");

    pybind11::class_<Policy, std::shared_ptr<Policy>, torch::nn::Module>(m, "Policy")
        .def(py::init())
        .def("forward", &Policy::forward);

    pybind11::class_<ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<size_t>())
        .def(py::init())
        .def("append", &ReplayBuffer::append_to_buffer, "append to buffer")
        .def("to_numpy", &ReplayBuffer::to_numpy, "convert buffer to numpy array");
    pybind11::class_<MDPTransition>(m, "MDPTransition")
        .def(py::init<Eigen::VectorXd, int, double>())
        .def("get", &MDPTransition::get);
}
