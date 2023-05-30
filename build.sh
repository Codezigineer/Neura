#!/bin/bash -e
set -e
mkdir build
asc ./src/*.ts --optimizeLevel 3 -o ./build/neuraFF.wasm
tsc ./src/js/*.ts --module es2015 --lib dom --target es2020 --outDir ./build 
