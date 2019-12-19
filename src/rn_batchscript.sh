#!/bin/bash
#SBATCH -n 1		# Number of tasks
#SBATCH -J resnet 	# Name of the job
#SBATCH -p p4		# Partition
#SBATCH -N 1            # number of nodes
#SBATCH -c 16
#SBATCH -o ./out_resnet.out
#SBATCH -e ./err_resnet.err

python resnet.py config_lesta_df.ini 
