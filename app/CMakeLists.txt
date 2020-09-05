add_executable(inspect_game inspect_game.cpp)
# add games to have them all registered before main
target_link_libraries(inspect_game deep_hedge_mccfr games)

add_executable(run_experiment run_experiment.cpp)
# add games to have them all registered before main
target_link_libraries(run_experiment deep_hedge_mccfr games)
