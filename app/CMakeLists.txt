add_executable(dmc_example dmc_example.cpp)
target_link_libraries(dmc_example deep_mw_cfr)

add_executable(inspect_game inspect_game.cpp)
# add games to have them all registered before main
target_link_libraries(inspect_game deep_mw_cfr games)
