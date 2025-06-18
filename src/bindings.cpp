// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>
// #include "include/pod.hpp"

// namespace py = pybind11;
// using namespace dealii;

// py::array_t<double> vector_to_numpy(const dealii::Vector<double> &vec) {
//     return py::array(vec.size(), vec.begin());
// }

// LAPACKFullMatrix<double> to_full_matrix(py::array_t<double> arr) {
//     auto buf = arr.request();
//     if (buf.ndim != 2)
//         throw std::runtime_error("Input array must be 2-dimensional.");

//     const size_t rows = buf.shape[0];
//     const size_t cols = buf.shape[1];
//     auto* ptr = static_cast<double*>(buf.ptr);

//     LAPACKFullMatrix<double> matrix(rows, cols);
//     for (size_t i = 0; i < rows; ++i)
//         for (size_t j = 0; j < cols; ++j)
//             matrix(i, j) = ptr[i * cols + j];  // row-major indexing
//     return matrix;
// }

// PYBIND11_MODULE(pod, m) {
//     py::class_<POD>(m, "POD")
//         .def(py::init<unsigned int, std::string &>())
//         .def("store_data", [](POD &self, py::array_t<double> arr) {
//             self.store_data(to_full_matrix(arr));
//         })
//         .def("get_singular_values", &POD::get_singular_values)
//         .def("fit", &POD::fit)
//         .def("get_snapshot_row", [](POD &self, unsigned int i) {
//             return vector_to_numpy(self.get_snapshot_row(i));
//         })
//         .def("get_snapshot_column", [](POD &self, unsigned int i) {
//             return vector_to_numpy(self.get_snapshot_column(i));
//         })
//         .def("predict", [](POD &self, std::vector<double> &params) {
//             return vector_to_numpy(self.predict(params));
//         })
//         ;
// }

#include "include/thermal_conduction.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<double> vector_to_numpy(const dealii::Vector<double> &vec) {
  return py::array_t<double>(vec.size(), vec.begin());
}

py::tuple sparse_matrix_to_numpy(const dealii::SparseMatrix<double> &matrix) {
  std::vector<unsigned int> rows;
  std::vector<unsigned int> cols;
  std::vector<double> values;

  for (size_t i = 0; i < matrix.m(); ++i) {
    for (auto it = matrix.begin(i); it != matrix.end(i); ++it) {
      rows.push_back(it->row());
      cols.push_back(it->column());
      values.push_back(it->value());
    }
  }

  // Return as a Python tuple of numpy arrays
  return py::make_tuple(py::array(rows.size(), rows.data()),
                        py::array(cols.size(), cols.data()),
                        py::array(values.size(), values.data()));
}

PYBIND11_MODULE(thermal_conduction_bindings, m) {
  py::class_<ThermalConduction>(m, "ThermalConduction")
      // .def(py::init<std::string &>())
      .def(py::init<
                   std::vector<double>,
                   std::vector<double>,
                   std::vector<double>,
                   unsigned int,
                   std::string>(),
               py::arg("regions"),
               py::arg("conductivities"),
               py::arg("boundary_temperatures") = std::vector<double>{0.0, 0.0, 500.0, -1.0},
               py::arg("axis") = 0,
               py::arg("parameter_file") = "params.prm")
      .def("compute_affine_decomposition",
           &ThermalConduction::compute_affine_decomposition)
      .def("get_rhs",
           [](ThermalConduction &self) {
             return vector_to_numpy(self.get_rhs());
           })
      .def("get_stiffness_matrix",
           [](ThermalConduction &self) {
             return sparse_matrix_to_numpy(self.get_stiffness_matrix());
           })
      .def("get_affine_stiffness_matrix",
           [](ThermalConduction &self, unsigned int i) {
             return sparse_matrix_to_numpy(self.get_affine_stiffness_matrix(i));
           })
      .def("get_affine_rhs",
           [](ThermalConduction &self, unsigned int i) {
             return vector_to_numpy(self.get_affine_rhs(i));
           })
      .def("get_num_regions", &ThermalConduction::get_num_regions)
      .def("get_x", &ThermalConduction::get_x)
      .def("get_y", &ThermalConduction::get_y)
      .def("set_regions", &ThermalConduction::set_regions)
      .def("set_conductivities", &ThermalConduction::set_conductivities)
      .def("set_boundary_temperatures", &ThermalConduction::set_boundary_temperatures)
      .def("run_assemble_system", &ThermalConduction::run_assemble_system)
      .def("get_system_matrix",
           [](ThermalConduction &self) {
             return sparse_matrix_to_numpy(self.get_system_matrix());
           })
      .def("get_rhs",
           [](ThermalConduction &self) {
             return vector_to_numpy(self.get_rhs());
           })
      .def("solve_system",
           [](ThermalConduction &self) {
             return vector_to_numpy(self.solve_system());
           })
  ;
}