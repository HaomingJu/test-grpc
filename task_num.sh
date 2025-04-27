#!/usr/bin/env bash

#curl http://123.57.18.145:8503/v1/models/num_1-yolo11n_saved_model
curl http://123.57.18.145:8503/v1/models/num_1-on-featurize

pushd install/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib
    ./bin/grpc-client \
        -t 123.57.18.145:8502 \
        -i ./bin/11866.png \
        -o ./bin/result_11866.png \
        --tensor-input-dim 1 640 640 3 \
        --tensor-output-dim 1 15 8400 \
        --model-name num_1-on-featurize
popd
