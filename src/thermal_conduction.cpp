#include "include/thermal_conduction.hpp"
#include "include/utils.hpp"
#include "deal.II/base/function.h"

ThermalConduction::ThermalConduction(std::vector<double> regions,
                        std::vector<double> conductivities,
                        std::vector<double> boundary_temperatures,
                        unsigned int axis,
                        std::string parameter_file)
    : grid_parameters(GridParameters())
    , triangulation()
    , fe(1)
    , dof_handler(triangulation)
    , regions(regions)
    , axis(axis)
    , conductivities(conductivities)
    , boundary_temperatures(boundary_temperatures)
{
    ParameterAcceptor::initialize(parameter_file);
    make_grid();
}

void ThermalConduction::make_grid() {
  if (grid_parameters.create_grid == true) {
    Point<dim> origin(0, 0);
    GridGenerator::hyper_rectangle(triangulation, origin, grid_parameters.point,
                                   true);
    triangulation.refine_global(grid_parameters.refinement);
  } else {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file(grid_parameters.grid_path);
    grid_in.read_vtu(input_file);
  }
  if (grid_parameters.save_grid == true) {
    std::ofstream out(grid_parameters.grid_path);
    GridOut grid_out;
    GridOutFlags::Vtu vtu_flags;
    vtu_flags.serialize_triangulation = true;
    grid_out.set_flags(vtu_flags);
    grid_out.write_vtu(triangulation, out);
  }
}

void ThermalConduction::setup_system() {
  dof_handler.distribute_dofs(fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  system_rhs.reinit(dof_handler.n_dofs());
}

void ThermalConduction::assemble_system_matrix(
    FEValues<dim> &fe_values, FullMatrix<double> &cell_matrix, int region) {
  for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        cell_matrix(i, j) += conductivities[region] * fe_values.shape_grad(i, q) *
                             fe_values.shape_grad(j, q) * fe_values.JxW(q);
}

void ThermalConduction::apply_boundary_conditions(bool apply_bc_to_solution) {
  std::map<types::global_dof_index, double> boundary_values;
  for (unsigned int i = 0; i < boundary_temperatures.size();
       i++) {
    const double temperature = boundary_temperatures[i];
    if (temperature >= 0) {
      Functions::ConstantFunction<dim> boundary_function(temperature);
      VectorTools::interpolate_boundary_values(
          dof_handler, types::boundary_id(i), boundary_function,
          boundary_values);
    }
  }
  apply_diagonal_boundary_conditions(boundary_values, apply_bc_to_solution);
}

void ThermalConduction::assemble_system(bool apply_bc_to_solution) {
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_JxW_values |
                              update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_cond(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    
    int region = get_point_region(cell->center());
    if (region != active_region && active_region != -1)
      continue;
    assemble_system_matrix(fe_values, cell_matrix, region);

    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                          cell_matrix(i, j));
  }
  apply_boundary_conditions(apply_bc_to_solution);
}

void ThermalConduction::compute_affine_decomposition(unsigned int i) {
    setup_system();
    active_region = i;
    system_matrix = 0;
    system_rhs = 0;
    assemble_system();
    active_region = -1;
}

SparseMatrix<double> &ThermalConduction::get_stiffness_matrix() {
  return system_matrix;
}

std::pair<Vector<double>, SparseMatrix<double>>
ThermalConduction::get_affine_components(unsigned int i) {
    compute_affine_decomposition(i);
    return std::make_pair(std::move(system_rhs), std::move(system_matrix));
}

unsigned int ThermalConduction::get_num_regions() {
  return regions.size() + 1;
}

std::vector<double> ThermalConduction::get_x() {
  std::map<types::global_dof_index, Point<dim>> support_points;
  std::vector<double> support_x(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler,
                                       support_points);
  unsigned int i = 0;
  for (const auto &pair : support_points) {
    const Point<dim> &point = pair.second;
    support_x[i] = point[0];
    ++i;
  }
  return support_x;
}

std::vector<double> ThermalConduction::get_y() {
  std::map<types::global_dof_index, Point<dim>> support_points;
  std::vector<double> support_y(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler,
                                       support_points);
  unsigned int i = 0;
  for (const auto &pair : support_points) {
    const Point<dim> &point = pair.second;
    support_y[i] = point[1];
    ++i;
  }
  return support_y;
}

int ThermalConduction::get_point_region(const Point<2> &point) {
  if (regions.empty())
    return 0;
  double y = point[axis];
  if (y > regions[regions.size() - 1])
    return regions.size();

  int index = 0;
  while (y > regions[index])
    index++;
  return index;
}

void ThermalConduction::apply_diagonal_boundary_conditions(
    std::map<types::global_dof_index, double> &boundary_values,
    bool apply_to_solution) {
  MappingQ1<dim> mapping;
  std::map<types::global_dof_index, Point<dim>> support_points;
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
  for (const auto &pair : boundary_values) {
    types::global_dof_index dof_index = pair.first;
    double rhs_value = pair.second;
    // Set the diagonal entry to 1 and all other entries in the row to 0
    double value = 0;
    for (typename SparseMatrix<double>::iterator p = system_matrix.begin(dof_index);
            p != system_matrix.end(dof_index); ++p)
    {
        value = p->value();
        p->value() = 0.;
    }

    if (get_point_region(support_points[dof_index]) == active_region || active_region == -1)  {
        
      system_matrix.set(dof_index, dof_index, value);
      system_rhs[dof_index] = rhs_value * value;
      if (apply_to_solution) 
        solution[dof_index] = rhs_value;
    }
  }
}

void ThermalConduction::set_regions(std::vector<double> regions) {
  this->regions = regions;
}

void ThermalConduction::set_conductivities(std::vector<double> conductivities) {
  this->conductivities = conductivities;
}

void ThermalConduction::set_boundary_temperatures(std::vector<double> boundary_temperatures) {
   this->boundary_temperatures = boundary_temperatures;
}

SparseMatrix<double> &ThermalConduction::get_system_matrix() {
  return system_matrix;
}

Vector<double> &ThermalConduction::get_rhs() {
  return system_rhs;
}

void ThermalConduction::run_assemble_system()
{
    setup_system();
    assemble_system();
}

Vector<double> ThermalConduction::solve_system() {
    setup_system();
    solution.reinit(dof_handler.n_dofs());
    assemble_system(true);
    
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());
    return solution;
}