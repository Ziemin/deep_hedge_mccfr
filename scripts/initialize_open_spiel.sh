#!/bin/bash

set -e
set -x

cd $(dirname $(dirname "$(realpath "$0")"))/extern/open-spiel

source "./open_spiel/scripts/global_variables.sh"

# For the external dependencies, we use fixed releases for the repositories that
# the OpenSpiel team do not control.
# Feel free to upgrade the version after having checked it works.
[[ -d "./pybind11" ]] || git clone -b 'v2.2.4' --single-branch --depth 1 https://github.com/pybind/pybind11.git
# The official https://github.com/dds-bridge/dds.git seems to not accept PR,
# so we have forked it.
[[ -d open_spiel/games/bridge/double_dummy_solver ]] || \
    git clone -b 'develop' --single-branch --depth 1 https://github.com/jblespiau/dds.git  \
        open_spiel/games/bridge/double_dummy_solver

if [[ ! -d open_spiel/abseil-cpp ]]; then
    git clone -b '20200225.1' --single-branch --depth 1 https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp
fi


# Optional dependencies.
DIR="open_spiel/games/hanabi/hanabi-learning-environment"
if [[ ${BUILD_WITH_HANABI:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
    git clone -b 'master' --single-branch --depth 15 https://github.com/deepmind/hanabi-learning-environment.git ${DIR}
    # We checkout a specific CL to prevent future breakage due to changes upstream
    # The repository is very infrequently updated, thus the last 15 commits should
    # be ok for a long time.
    pushd ${DIR}
    git checkout  'b31c973'
    popd
fi


# This Github repository contains the raw code from the ACPC server
# http://www.computerpokercompetition.org/downloads/code/competition_server/project_acpc_server_v1.0.42.tar.bz2
# with the code compiled as C++ within a namespace.
DIR="open_spiel/games/universal_poker/acpc"
if [[ ${BUILD_WITH_ACPC:-"ON"} == "ON" ]] && [[ ! -d ${DIR} ]]; then
    git clone -b 'master' --single-branch --depth 1  https://github.com/jblespiau/project_acpc_server.git ${DIR}
fi
