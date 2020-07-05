file(GLOB DEEP_MW_CFR_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/dmc/*.hpp")
file(GLOB DEEP_MW_CFR_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_library(
  deep_mw_cfr
  STATIC
  ${DEEP_MW_CFR_SOURCES}
  ${DEEP_MW_CFR_HEADERS}
  ${OPEN_SPIEL_OBJECTS}
)
add_library(deep_mw_cfr::deep_mw_cfr ALIAS deep_mw_cfr)
target_compile_features(deep_mw_cfr PUBLIC cxx_std_20)
target_compile_options(deep_mw_cfr PRIVATE -Wall -Wextra -Wunreachable-code)

target_include_directories(
  deep_mw_cfr
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  ${PROJECT_SOURCE_DIR}/extern/open-spiel
  ${PROJECT_SOURCE_DIR}/extern/open-spiel/open_spiel/abseil-cpp
  ${TORCH_INCLUDE_DIRS}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(deep_mw_cfr
  PUBLIC
  ${TORCH_LIBRARIES}
  Boost::filesystem
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
