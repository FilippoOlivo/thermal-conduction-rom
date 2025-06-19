#include <deal.II/base/config.h>

#include "deal.II/base/index_set.h"
#include <deal.II/base/hdf5.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include "deal.II/lac/sparsity_pattern.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <ios>
#include <iostream>

#include "parameter_acceptor.hpp"
using namespace dealii;

static inline constexpr unsigned int dim = 2;

class ThermalConduction
{
  public:
    ThermalConduction(std::string &parameter_file);
    
    ThermalConduction(std::vector<double> regions,
                      std::vector<double> conductivities,
                      std::vector<double> boundary_temperatures = {0.0, 0.0, 500.0, -1.0},
                      unsigned int axis = 0,
                      std::string parameter_file = "params.prm");
    
    void 
    compute_stiffness_matrix();
    
    SparseMatrix<double>&
    get_stiffness_matrix();
    
    std::pair<Vector<double>, SparseMatrix<double>>
    get_affine_components(unsigned int i);
    
    unsigned int
    get_num_regions();
    
    std::vector<double>
    get_y();
    
    std::vector<double>
    get_x();
    
    void set_conductivities(std::vector<double> conductivities);
    
    void set_regions(std::vector<double> regions);
    
    void set_boundary_temperatures(std::vector<double> boundary_temperatures);
    
    void set_axis(unsigned int axis);
    
    SparseMatrix<double>& get_system_matrix();
    
    Vector<double>& get_rhs();
    
    void
    setup_system();
    
    void
    run_assemble_system();
    
    Vector<double>
    solve_system();
    
    GridParameters            grid_parameters;
    Triangulation<dim>        triangulation;
    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;
    std::vector<double>       regions;
    unsigned int            axis;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;
    Vector<double>            system_rhs;
    std::vector<SparseMatrix<double>> affine_stiffness_matrices;
    std::vector<Vector<double>> affine_rhs;
    int            active_region = -1;
    std::vector<double> conductivities;
    std::vector<double> boundary_temperatures;
    Vector<double> y;
    Vector<double> solution;


  private:
    void
    make_grid();
    
    void
    assemble_system(bool apply_bc_to_solution=false);

    void
    apply_boundary_conditions(bool apply_to_solution=false);
    
    void
    assemble_system_matrix(FEValues<dim>      &fe_values,
                           FullMatrix<double> &cell_matrix,
                           int region);
    
    void
    assemble_system_rhs(
        FEFaceValues<dim>                                    &fe_face_values,
        Vector<double>                                       &cell_rhs,
        const typename DoFHandler<dim>::active_cell_iterator &cell);
     
    int 
    get_point_region(const Point<2> &point);
    
    void
    apply_diagonal_boundary_conditions(std::map<types::global_dof_index, double> &boundary_values, bool apply_to_solution=false);
    
    void compute_affine_decomposition(unsigned int i);
    
};
