#!/usr/bin/env bash

#curl http://123.57.18.145:8501/v1/models/dogcat

curl http://123.57.18.145:8501/v1/models/dogcat2-on-featurize


pushd install/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib
    ./bin/grpc-client \
        -t 123.57.18.145:8500 \
        -i ./bin/cat.jpg \
        -o ./bin/result_cat.jpg \
        --tensor-input-dim 1 640 640 3 \
        --tensor-output-dim 1 6 8400 \
        --model-name dogcat2-on-featurize
popd
