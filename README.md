# Deep VR-MCCFR

## Requirements
* Pytorch (libtorch) - installation instructions [here](https://github.com/pytorch/pytorch#install-pytorch).

## Installation

```bash
mkdir build
cd build
cmake  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda -DCMAKE_PREFIX_PATH=<torch_installation_path>/share/cmake ..
```
