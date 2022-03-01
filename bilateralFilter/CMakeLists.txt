cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(ParticleReader)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkCommonCore
  vtkIOGeometry
  vtkInteractionStyle
  vtkRenderingContextOpenGL2
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingGL2PSOpenGL2
  vtkRenderingOpenGL2
  QUIET
)

if (NOT VTK_FOUND)
  message("Skipping ParticleReader: ${VTK_NOT_FOUND_MESSAGE}")
  return()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  add_executable(ParticleReader MACOSX_BUNDLE ParticleReader.cxx )
  target_link_libraries(ParticleReader PRIVATE ${VTK_LIBRARIES})
else()
  # Prevent a "command line is too long" failure in Windows.
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")
  add_executable(ParticleReader MACOSX_BUNDLE ascread.cpp )
  target_link_libraries(ParticleReader PRIVATE ${VTK_LIBRARIES})
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS ParticleReader
    MODULES ${VTK_LIBRARIES}
    )
endif()
