#!/usr/bin/bash
#get_dataset.sh

CWD=$(dirname $0)
DATA=$CWD/../data/
echo Getting train data into $DATA
wget -P $DATA http://metcalf1.difa.unibo.it/DATA3/datapack2.0train.tar.gz
tar xvzf datapack2.0train.tar.gz
echo Getting test data into $DATA
wget -P $DATA http://metcalf1.difa.unibo.it/DATA3/datapack2.0test.tar.gz
tar xvzf datapack2.0test.tar.gz
