#!/bin/bash 

lensdir=/home/epfl/esavary/Documents/Euclid_challenge/Lenses
nolensdir=/home/epfl/esavary/Documents/Euclid_challenge/NonLenses
datadir=/home/epfl/dforero/gravitational_lens_ml/data/elodiedata
catalogdir=$datadir/../catalog
traincat=$catalogdir/newdata_catalog.csv
targetdir=$datadir/all
nolensvaldir=$valdir/nolens
nlenses=$(ls $lensdir/ | wc -l)
rm -v -r $targetdir/*
mkdir -v -p $targetdir
nval=$(($nlenses / 3))
ntrain=$(($nlenses - $nval))
echo $nlenses $nval $ntrain
echo "filename,label" > $traincat
iter=0
for filename in $(ls -d "$lensdir"/* | shuf | grep fits)
do
  if [[ $iter -le $nlenses ]]; then
   # ./fits2npy.py $filename $targetdir
    echo $filename, 1 >> $traincat
  else
    break
  fi
  iter=$(($iter + 1))
done

iter=0
for filename in $(ls -d "$nolensdir"/* | shuf | grep fits)
do
  if [[ $iter -le $nlenses ]]; then
   # ./fits2npy.py $filename $targetdir
    echo $filename, 0 >> $traincat
  else
    break
  fi
  iter=$(($iter + 1))
done
