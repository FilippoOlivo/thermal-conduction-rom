#include <deal.II/base/config.h>
#include <deal.II/base/parameter_acceptor.h>

using namespace dealii;

class GridParameters : public ParameterAcceptor {
public:
  GridParameters() : ParameterAcceptor("Grid") {
    add_parameter("create_grid", create_grid, "Create grid");
    add_parameter("refinement", refinement, "Length of the domain");
    add_parameter("point", point, "Point to build rectangular mesh");
    add_parameter("save_grid", save_grid, "Whether to save the created grid");
    add_parameter("grid_path", grid_path, "Path to the grid file");
  }

  bool create_grid = true;
  unsigned int refinement = 5;
  Point<2> point = Point<2>(1.0, 1.0);
  bool save_grid = false;
  std::string grid_path = "grid.msh";
};