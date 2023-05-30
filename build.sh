#!/bin/bash -e
set -e
mkdir build
npm run asc ./src/*.ts --optimizeLevel 3 -o ./build neuraFF.wasm
npm run tsc ./src/js/*.ts --outDir ./build
