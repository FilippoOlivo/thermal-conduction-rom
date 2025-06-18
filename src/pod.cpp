#include "include/pod.hpp"
#include <algorithm>
#include <deal.II/lac/precondition.h>
#include <filesystem>
#include <iostream>

LAPACKFullMatrix<double> to_matrix(std::vector<Vector<double>> &vec) {
  LAPACKFullMatrix<double> mat(vec[0].size(), vec.size());
  for (unsigned int i = 0; i < vec.size(); ++i)
    for (unsigned int j = 0; j < vec[i].size(); ++j)
      mat(j, i) = vec[i][j];
  return mat;
}

Vector<double> get_row(LAPACKFullMatrix<double> &mat,
                       const unsigned int row_idx) {
  unsigned int n = mat.n();
  Vector<double> row(n);
  for (unsigned int i = 0; i < n; i++)
    row[i] = mat(row_idx, i);
  return row;
}

Vector<double> get_column(LAPACKFullMatrix<double> &mat,
                          const unsigned int column_idx) {
  unsigned int m = mat.m();
  Vector<double> column(m);
  for (unsigned int i = 0; i < m; i++)
    column[i] = mat(i, column_idx);
  return column;
}

POD::POD(const unsigned int n_modes, std::string params_path)
    : n_modes(n_modes), problem(params_path),
      stiffness_matrices(problem.get_affine_stiffness_matrix()),
      rhs(problem.get_rhs()) {}

void POD::fit() {
  compute_pod_basis();
  compute_reduced_stiffness_matrices();
  compute_reduced_rhs();
}

void POD::compute_pod_basis() {
  mat.compute_svd();
  unsigned int n_singular_values = std::min(mat.m(), mat.n());
  singular_values.resize(n_singular_values);
  for (unsigned int i = 0; i < n_singular_values; i++)
    singular_values[i] = mat.singular_value(i);
  LAPACKFullMatrix<double> pod_modes_matrix = mat.get_svd_u();
  pod_modes.resize(this->n_modes);
  std::cout << "POD modes size: " << pod_modes_matrix.m() << " x "
            << pod_modes_matrix.n() << std::endl;
  for (unsigned int i = 0; i < this->n_modes; ++i)
    pod_modes[i] = get_column(pod_modes_matrix, i);
}

void POD::store_data(LAPACKFullMatrix<double> matrix) { mat = matrix; }

std::vector<double> POD::get_singular_values() { return singular_values; }

Vector<double> POD::get_snapshot_row(unsigned int row_index) {
  unsigned int n = mat.n();
  Vector<double> row(n);
  for (unsigned int i = 0; i < n; i++)
    row[i] = mat(row_index, i);
  return row;
}

Vector<double> POD::get_snapshot_column(unsigned int column_index) {
  unsigned int m = mat.m();
  Vector<double> column(m);
  for (unsigned int i = 0; i < m; i++)
    column[i] = mat(i, column_index);
  return column;
}

void POD::compute_reduced_stiffness_matrices() {
  const unsigned int n_terms = stiffness_matrices.size();
  reduced_stiffness_matrices.resize(n_terms);

  Vector<double> temp(rhs.size());

  for (unsigned int mat = 0; mat < n_terms; ++mat) {
    reduced_stiffness_matrices[mat].reinit(n_modes, n_modes);
    reduced_stiffness_matrices[mat] = 0.0;

    for (unsigned int j = 0; j < n_modes; ++j) {
      stiffness_matrices[mat].vmult(temp, pod_modes[j]);

      for (unsigned int i = 0; i < n_modes; ++i)

        reduced_stiffness_matrices[mat](i, j) += pod_modes[i] * temp;
    }
  }
}

void POD::compute_reduced_rhs() {
  reduced_rhs.reinit(n_modes);
  reduced_rhs = 0.0;
  for (unsigned int i = 0; i < rhs.size(); i++)
    if (rhs[i] > 0.0)
      std::cout << "rhs[" << i << "] = " << rhs[i] << std::endl;

  for (unsigned int i = 0; i < n_modes; ++i)
    reduced_rhs[i] = pod_modes[i] * rhs;
}

LAPACKFullMatrix<double> POD::apply_parameters(std::vector<double> &params) {
  unsigned int n_terms = stiffness_matrices.size();
  LAPACKFullMatrix<double> system_matrix(n_modes, n_modes);
  system_matrix = 0.0;
  for (unsigned int mat = 0; mat < n_terms; ++mat)
    for (unsigned int i = 0; i < n_modes; ++i)
      for (unsigned int j = 0; j < n_modes; ++j)
        system_matrix(i, j) +=
            params[mat] * reduced_stiffness_matrices[mat](i, j);
  return system_matrix;
}

Vector<double> POD::predict(std::vector<double> params) {

  LAPACKFullMatrix<double> system_matrix = apply_parameters(params);

  Vector<double> reduced_solution(n_modes);

  system_matrix.compute_lu_factorization();
  system_matrix.solve(reduced_rhs);
  reduced_solution = reduced_rhs;

  // Compute residual
  Vector<double> to_compute_res(n_modes);
  system_matrix.vmult(to_compute_res, reduced_solution);
  Vector<double> res = to_compute_res;
  res -= reduced_rhs;
  std::cout << "Residual norm: " << res.l2_norm() << std::endl;

  // Reconstruct full solution
  Vector<double> full_solution(rhs.size());
  for (unsigned int i = 0; i < n_modes; ++i)
    full_solution.add(reduced_solution[i], pod_modes[i]);
  return full_solution;
}
