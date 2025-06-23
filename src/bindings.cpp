#include "include/thermal_conduction.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace dealii;

py::array_t<double> vector_to_numpy(const Vector<double> &vec) {
  return py::array_t<double>(vec.size(), vec.begin());
}

py::tuple sparse_matrix_to_numpy(const SparseMatrix<double> &matrix) {
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
  return py::make_tuple(py::array(rows.size(), rows.data()),
                        py::array(cols.size(), cols.data()),
                        py::array(values.size(), values.data()));
}

py::tuple convert_affine_decomposition(
    std::pair<Vector<double>, SparseMatrix<double>> affine_components) {
  py::array_t<double> system_rhs = vector_to_numpy(affine_components.first);
  py::tuple system_matrix = sparse_matrix_to_numpy(affine_components.second);
  py::tuple to_return = py::make_tuple(system_rhs, system_matrix);
  return to_return;
}

PYBIND11_MODULE(thermal_conduction, m) {
  py::class_<ThermalConduction>(m, "ThermalConduction")
      .def(py::init<std::vector<double>, std::vector<double>,
                    std::vector<double>, unsigned int, std::string>(),
           py::arg("regions"), py::arg("conductivities"),
           py::arg("boundary_temperatures") =
               std::vector<double>{0.0, 0.0, 500.0, -1.0},
           py::arg("axis") = 0, py::arg("parameter_file") = "params.prm")
      .def("get_rhs",
           [](ThermalConduction &self) {
             return vector_to_numpy(self.get_rhs());
           })
      .def("get_stiffness_matrix",
           [](ThermalConduction &self) {
             return sparse_matrix_to_numpy(self.get_stiffness_matrix());
           })
      .def("get_affine_components",
           [](ThermalConduction &self, unsigned int i) {
             return convert_affine_decomposition(self.get_affine_components(i));
           })
      .def("get_num_regions", &ThermalConduction::get_num_regions)
      .def("get_x", &ThermalConduction::get_x)
      .def("get_y", &ThermalConduction::get_y)
      .def("set_regions", &ThermalConduction::set_regions)
      .def("set_conductivities", &ThermalConduction::set_conductivities)
      .def("set_boundary_temperatures",
           &ThermalConduction::set_boundary_temperatures)
      .def("set_axis", &ThermalConduction::set_axis)
      .def("run_assemble_system", &ThermalConduction::run_assemble_system)
      .def("get_system_matrix",
           [](ThermalConduction &self) {
             return sparse_matrix_to_numpy(self.get_system_matrix());
           })
      .def("solve_system", [](ThermalConduction &self) {
        return vector_to_numpy(self.solve_system());
      });
}