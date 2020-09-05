file(GLOB DEEP_HEDGE_MCCFR_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/dhc/*.hpp")
file(GLOB DEEP_HEDGE_MCCFR_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_library(
  deep_hedge_mccfr
  STATIC
  ${DEEP_HEDGE_MCCFR_SOURCES}
  ${DEEP_HEDGE_MCCFR_HEADERS}
  ${OPEN_SPIEL_OBJECTS}
)
add_library(deep_hedge_mccfr::deep_hedge_mccfr ALIAS deep_hedge_mccfr)
target_compile_features(deep_hedge_mccfr PUBLIC cxx_std_20)
target_compile_options(deep_hedge_mccfr PRIVATE -Wall -Wextra -Wunreachable-code)

target_include_directories(
  deep_hedge_mccfr
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  ${PROJECT_SOURCE_DIR}/extern/open-spiel
  ${PROJECT_SOURCE_DIR}/extern/open-spiel/open_spiel/abseil-cpp
  ${TORCH_INCLUDE_DIRS}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(deep_hedge_mccfr
  PUBLIC
  ${TORCH_LIBRARIES}
  Boost::filesystem
  Boost::serialization
  fmt::fmt
  range-v3
  nlohmann_json::nlohmann_json
  absl::container
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::flat_hash_map
  absl::optional
  absl::str_format
  absl::strings
  absl::time)
