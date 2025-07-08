#!/bin/bash
for dir in experiments/pod_nn/* ; do
    echo "Processing $dir"
    python run.py --config $dir
done
for dir in experiments/pod_dl_rom/* ; do
    echo "Processing $dir"
    python run.py --config $dir
done
for dir in experiments/dl_rom/* ; do
    echo "Processing $dir"
    python run.py --config $dir
done

python run.py --config experiments/fno.yaml