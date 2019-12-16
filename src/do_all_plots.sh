#!/usr/bin/bash
if [[ $# -ne 2 ]]; then
echo ERROR: Unexpected number of arguments
echo USAGE: $0 WORKDIR CONFIG_FILE
exit 1
fi
dir=$1
config=$2
for filename in $(ls -p $1/results/checkpoints/lastro_cnn/*.h5); do
python plots_from_dat.py $1/src/$2 $filename
done
for filename in $(ls -p $1/results/lastro_cnn*.h5); do
python plots_from_dat.py $1/src/$2 $filename
done
