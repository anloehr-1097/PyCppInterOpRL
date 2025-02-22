#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>

namespace py = pybind11;
using namespace pybind11::literals;


class TArray {

private:
    size_t sz;
    float *ar_data;

public: 
    TArray(size_t num_entries): sz(num_entries){
        ar_data = new float[sz];
    };
};

int main() {
    py::module_ gym = py::module_::import("gymnasium");
    py::object env = gym.attr("make")("LunarLander-v3", "render_mode"_a="human");

    



}
