cmake_minimum_required(VERSION 3.14)
project(deep_mw_cfr VERSION 0.0.1 LANGUAGES CXX)

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

# FMT
add_subdirectory(extern/fmt EXCLUDE_FROM_ALL)

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
add_subdirectory(extern/open-spiel/open_spiel EXCLUDE_FROM_ALL)

set (OPEN_SPIEL_OBJECTS
  $<TARGET_OBJECTS:open_spiel_core>
  $<TARGET_OBJECTS:games>
  $<TARGET_OBJECTS:game_transforms>
  $<TARGET_OBJECTS:open_spiel_query>
  $<TARGET_OBJECTS:bridge_double_dummy_solver>
  $<TARGET_OBJECTS:algorithms>
  $<TARGET_OBJECTS:utils>
  )
if (BUILD_WITH_HANABI)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:hanabi_learning_environment>)
endif()
if (BUILD_WITH_ACPC)
  set(OPEN_SPIEL_OBJECTS ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:universal_poker_clib> $<TARGET_OBJECTS:universal_poker_lib>)
endif()


###############################################################

# clang tidy
# set(CMAKE_CXX_CLANG_TIDY
#   clang-tidy;
#   -header-filter=.;
#   -checks=*)

# set(CMAKE_CXX_CPPCHECK "cppcheck")

add_subdirectory(src)
add_subdirectory(app)

option(DEEP_MW_CFR_BUILD_TESTS "Should the tests for Deep-MW-CFR be built" ON)
enable_testing()
# add_subdirectory(tests)
