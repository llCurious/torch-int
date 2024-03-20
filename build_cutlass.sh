export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd submodules/cutlass
rm -rf build
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=86 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 16
