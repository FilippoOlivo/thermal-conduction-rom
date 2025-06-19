# cmake_minimum_required(VERSION 3.18)
# project(POD LANGUAGES CXX)

# # Avoid Python warnings from pybind11
# if(POLICY CMP0148)
#   cmake_policy(SET CMP0148 NEW)
# endif()

# # Required packages
# find_package(deal.II REQUIRED HINTS $ENV{DEAL_II_DIR})
# find_package(pybind11 REQUIRED)


# # Build Python module
# add_library(pod MODULE
#   src/bindings.cpp
#   src/pod.cpp
#   src/thermal_conduction.cpp
# )

# # Output as Python-style shared object (no lib prefix)
# set_target_properties(pod PROPERTIES
#   PREFIX ""
#   OUTPUT_NAME "pod"
# )

# # Link libraries (plain syntax for compatibility with deal.II macros)
# target_link_libraries(pod
#   pybind11::module
#   ${DEAL_II_LIBRARIES}
# )

# # Apply deal.II flags/macros
# deal_ii_setup_target(pod RELEASE)

# # ----------------------------
cmake_minimum_required(VERSION 3.18)
project(ThermalConduction LANGUAGES CXX)

# Enable CMP0148 to avoid Python warnings from pybind11
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Deal.II and pybind11
find_package(deal.II REQUIRED HINTS $ENV{DEAL_II_DIR})
find_package(pybind11 REQUIRED)

# Create module
add_library(thermal_conduction MODULE
  src/bindings.cpp
  src/thermal_conduction.cpp
)

# Use deal.II macros and link against pybind11
deal_ii_setup_target(thermal_conduction RELEASE)
target_link_libraries(thermal_conduction pybind11::module)

# Output Python-style shared object (no 'lib' prefix)
set_target_properties(thermal_conduction PROPERTIES
  PREFIX ""
  OUTPUT_NAME "thermal_conduction"
)