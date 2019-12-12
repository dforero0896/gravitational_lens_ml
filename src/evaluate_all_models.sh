#!/usr/bin/bash
if [[ $# -ne 1 ]]; then
echo ERROR: Provide the work directory please.
exit 1
fi
dir=$1
for filename in $(ls -p $1/results/checkpoints/lastro_cnn/*.h5); do
python analysis_lastro.py $1/src/config_lesta_df.ini $filename
done
for filename in $(ls -p $1/results/lastro_cnn*.h5); do
python analysis_lastro.py $1/src/config_lesta_df.ini $filename
done
