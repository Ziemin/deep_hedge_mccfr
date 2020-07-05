add_executable(inspect_game inspect_game.cpp)
# add games to have them all registered before main
target_link_libraries(inspect_game deep_mw_cfr games)

add_executable(dmc_experiment dmc_experiment.cpp)
# add games to have them all registered before main
target_link_libraries(dmc_experiment deep_mw_cfr games)
