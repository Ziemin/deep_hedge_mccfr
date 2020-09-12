# Deep-Hedge MCCFR

## Requirements
* Pytorch (libtorch) - installation instructions [here](https://github.com/pytorch/pytorch#install-pytorch).
* Boost >= 1.70 (filestem, serialization)

## Installation

Get the dependencies:
```bash
git submodule update --init --recursive 
```

Build the project:
```bash
mkdir build
cd build
cmake  -DCMAKE_BUILD_TYE=Release -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda -DCMAKE_PREFIX_PATH=<torch_installation_path>/share/cmake ..
```

## Running an experiment
```bash
./build/app/run_experiment --config ./configs/<config_name>.json --eval_freq 250 --dir ./experiments --name <experiment_name>
```

## Resuming an experiment
```bash
./build/app/run_experiment --checkpoint ./experiments/<experiment_name>/<experiment_time>/checkpoints --eval_freq 250 --dir ./experiments --name <experiment_name>
```
