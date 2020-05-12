cmake_minimum_required(VERSION 3.14)
project(deep_vr_mccfr VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################
# Dependencies

# Boost
set(Boost_USE_STTIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)
message(STATUS "Boost version: ${Boost_VERSION}")

# Torch
find_package(Torch REQUIRED)
message(STATUS "Torch version: ${TORCH_VERSION}")

# Catch2
set(CATCH_BUILD_TESTING OFF)
add_subdirectory(extern/Catch2 EXCLUDE_FROM_ALL)

# Ranges
add_subdirectory(extern/range-v3 EXCLUDE_FROM_ALL)

# Open-Spiel
option(BUILD_WITH_HANABI "Should Hanabi be downloaded and built" ON)
option(BUILD_WITH_ACPC "Should ACPC be downloaded and built" ON)

if (BUILD_WITH_HANABI)
  set(ENV{BUILD_WITH_HANABI} ${BUILD_WITH_HANABI})
endif ()
if (BUILD_WITH_ACPC)
  set(ENV{BUILD_WITH_ACPC} ${BUILD_WITH_ACPC})
endif ()
message("Initializing OpenSpiel's dependencies")
execute_process(
  COMMAND sh ./scripts/initialize_open_spiel.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE OPEN_SPIEL_DEPS)
if (NOT OPEN_SPIEL_DEPS EQUAL "0")
  message(FATAL_ERROR "./scripts/initialize_open_spiel.sh failed")
endif ()
add_subdirectory(extern/open-spiel/open_spiel)


###############################################################
# AREL

# clang tidy
# set(CMAKE_CXX_CLANG_TIDY
#   clang-tidy;
#   -header-filter=.;
#   -checks=*)

# set(CMAKE_CXX_CPPCHECK "cppcheck")

add_library(deep_vr_mccfr STATIC)
add_library(deep_vr_mccfr::deep_vr_mccfr ALIAS deep_vr_mccfr)
target_compile_features(deep_vr_mccfr PRIVATE cxx_std_20)
target_compile_options(deep_vr_mccfr PRIVATE -Wall -Wextra -Wunreachable-code)

target_include_directories(deep_vr_mccfr
  PUBLIC
    include
    # $<INSTALL_INTERFACE:include>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${TORCH_INCLUDE_DIRS}
  # PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(deep_vr_mccfr
  PUBLIC
    ${TORCH_LIBRARIES}
    Boost::filesystem
    range-v3)

option(AREL_BUILD_TESTS "Should the tests for Deep-VR-MCCFR be built" ON)

add_subdirectory(src)

enable_testing()
# add_subdirectory(tests)
