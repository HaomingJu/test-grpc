#!/usr/bin/env bash

rm build -rf

cmake -B build

cmake --build ./build --config Debug --target install -- -j32
