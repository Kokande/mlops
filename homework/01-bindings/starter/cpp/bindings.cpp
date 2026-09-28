#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    // TODO 2: validate dtype, ndim, matching shapes, contiguous/aligned buffers.
    // Allocate an independent output, call mac_kernel, and return the array.
    // The Python contract and exception types are specified in ../README.md.
    
    // dtype check
    if (
        !a.dtype().equal(py::dtype::of<double>()) ||
        !b.dtype().equal(py::dtype::of<double>()) ||
        !c.dtype().equal(py::dtype::of<double>())
    ) {
        throw py::type_error("a, b and c values should be native-endian (float64)");
    }

    // ndim and shape check
    if (
        a.ndim() != 3 || b.ndim() != 3|| c.ndim() != 3
    ) {
        throw py::value_error("Arrays should be 3D");
    }

    if (
        a.shape(0) != b.shape(0) || a.shape(0) != c.shape(0) ||
        a.shape(1) != b.shape(1) || a.shape(1) != c.shape(1) ||
        a.shape(2) != b.shape(2) || a.shape(2) != c.shape(2)
    ) {
        throw py::value_error("Array shapes do no match");
    }

    // noncontigious and unaligned check
    if (!(
        (a.flags() & py::array::c_style) && 
        (b.flags() & py::array::c_style) &&
        (c.flags() & py::array::c_style)
    )) {
        throw py::value_error("Values should have contigious view");
    }

    if (!(
        a.attr("flags").attr("aligned").cast<bool>() && 
        b.attr("flags").attr("aligned").cast<bool>() &&
        c.attr("flags").attr("aligned").cast<bool>()
    )) {
        throw py::value_error("Buffer should aligned");
    }

    // creating output
    py::buffer_info info = a.request();

    py::array_t<double> out(info.shape);
    std::fill(out.mutable_data(), out.mutable_data() + out.size(), 0.0);
    
    mac_kernel(
        static_cast<const double*>(a.data()), 
        static_cast<const double*>(b.data()), 
        static_cast<const double*>(c.data()), 
        out.mutable_data(), out.size()
    );

    return out;

    throw std::logic_error("TODO 2: implement array binding");
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    // TODO 2: expose mac(a, b, c). Do not allow implicit argument conversions.
    module.def("mac", &mac, py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
