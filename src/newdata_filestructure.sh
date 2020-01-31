#!/bin/bash 

lensdir=/home/epfl/esavary/Documents/Euclid_challenge/Lenses
nolensdir=/home/epfl/esavary/Documents/Euclid_challenge/NonLenses
datadir=/home/epfl/dforero/gravitational_lens_ml/data/elodiedata
catalogdir=$datadir/../catalog
traincat=$catalogdir/newdata_train.csv
valcat=$catalogdir/newdata_val.csv
traindir=$datadir/train
lenstraindir=$traindir/lens
nolenstraindir=$traindir/nolens
valdir=$datadir/val
lensvaldir=$valdir/lens
nolensvaldir=$valdir/nolens
nlenses=$(ls $lensdir/ | wc -l)
rm -v -r $traindir/*
rm -v -r $valdir/*
mkdir -v -p $lenstraindir
mkdir -v -p $lensvaldir
mkdir -v -p $nolenstraindir
mkdir -v -p $nolensvaldir
nval=$(($nlenses / 3))
ntrain=$(($nlenses - $nval))
echo $nlenses $nval $ntrain
echo "# Filename, label" > $traincat
echo "# Filename, label" > $valcat
iter=0
for filename in $(ls -d "$lensdir"/* | shuf )
do
  if [[ $iter -lt $ntrain ]]; then
    ./fits2png.py $filename $lenstraindir
    echo $filename 1 >> $traincat
  elif [[ $iter -gt $nlenses ]]; then
    break
  else
    ./fits2png.py $filename $lensvaldir
    echo $filename 1 >> $valcat
  fi
  iter=$(($iter + 1))
done

iter=0
for filename in $(ls -d "$nolensdir"/* | shuf )
do
  if [[ $iter -lt $ntrain ]]; then
    ./fits2png.py $filename $nolenstraindir
    echo $filename 0 >> $traincat
  elif [[ $iter -gt $nlenses ]]; then
    break
  else
    ./fits2png.py $filename $nolensvaldir
    echo $filename 0 >> $valcat
  fi
  iter=$(($iter + 1))
done
