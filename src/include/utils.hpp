#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

template <class MatrixType, class VectorType>
void
apply_diagonal_boundary_conditions(
    MatrixType                                &system_matrix,
    VectorType                                &system_rhs,
    std::map<types::global_dof_index, double> &boundary_values)
{
    for (const auto &pair : boundary_values)
        {
            types::global_dof_index dof_index = pair.first;
            double                  rhs_value = pair.second;
            // Set the diagonal entry to 1 and all other entries in the row to 0
            for (typename SparseMatrix<double>::iterator p =
                     system_matrix.begin(dof_index);
                 p != system_matrix.end(dof_index);
                 ++p)
                if (p->column() != dof_index)
                    p->value() = 0.;
            // Set the diagonal entry to 1 and the right-hand side value
            system_matrix.set(dof_index, dof_index, 1);
            system_rhs[dof_index] = rhs_value;
        }
}
