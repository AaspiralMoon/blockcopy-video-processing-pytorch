#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

torch::Tensor OBDS(torch::Tensor img) {
    return img;
}

PYBIND11_MODULE(OBDS_zoo, m) {
    m.def("OBDS", &OBDS, "Object-based Diamond Search");
}
