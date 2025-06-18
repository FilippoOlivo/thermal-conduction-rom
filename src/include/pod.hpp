#include "deal.II/lac/sparse_matrix.h"
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include "thermal_conduction.hpp"
#include <deal.II/lac/solver_cg.h>
#include <vector>

using namespace dealii;

class POD
{
    public:
        POD(const unsigned int n_modes, std::string params_path);
        void store_data(LAPACKFullMatrix<double> matrix);
        void set_stiffness_matrix(std::vector<SparseMatrix<double>>);
        std::vector<double> get_singular_values();
        void fit();
        Vector<double> get_snapshot_row(unsigned int row_index);
        Vector<double> get_snapshot_column(unsigned int column_index);
        Vector<double> predict(const std::vector<double> params);
        
        
    private:
        unsigned int n_modes; // Number of modes
        ThermalConduction problem; // Thermal conduction problem instance
        LAPACKFullMatrix<double> mat; // Snapshots matrix
        std::vector<double> singular_values; // Singular values
        std::vector<size_t> sv_indices; // Indices of the modes
        std::vector<Vector<double>> pod_modes; // POD basis vectors
        std::vector<SparseMatrix<double>>& stiffness_matrices; // Stiffness matricess
        std::vector<LAPACKFullMatrix<double>> reduced_stiffness_matrices; // Reduced stiffness matrices
        Vector<double> reduced_rhs; // Reduced right-hand side vector
        Vector<double>& rhs;
        void compute_pod_basis();
        void compute_reduced_stiffness_matrices();
        void compute_reduced_rhs();
        LAPACKFullMatrix<double> apply_parameters(std::vector<double>& params);
        
};