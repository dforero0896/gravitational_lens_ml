#!/usr/bin/bash
#get_dataset.sh

CWD=$(dirname $0)
DATA=$CWD/../data/
echo Getting train data into $DATA
if [[ ! -e $DATA/datapack2.0train.tar.gz ]];then
wget -P $DATA http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz
fi
if [[ ! -e $DATA/datapack2.0train ]]; then
mkdir -v $DATA/datapack2.0train
fi
tar xvzf $DATA/datapack2.0train.tar.gz -C $DATA/datapack2.0train
echo Getting test data into $DATA
if [[ ! -e $DATA/datapack2.0test.tar.gz ]];then
wget -P $DATA http://metcalf1.difa.unibo.it/DATA3/datapack2.0test.tar.gz
fi
if [[ ! -e $DATA/datapack2.0test ]]; then
mkdir -v $DATA/datapack2.0test
fi
tar xvzf $DATA/datapack2.0test.tar.gz -C $DATA/datapack2.0test
wget -P $DATA/catalog/ http://metcalf1.difa.unibo.it/DATA3/image_catalog2.0train.csv
