#!/bin/bash
#SBATCH -n 1		# Number of tasks
#SBATCH -J resnet 	# Name of the job
#SBATCH -p gpu		# Partition
#SBATCH -N 1            # number of nodes
#SBATCH -o ./out_resnet.out
#SBATCH -e ./err_resnet.err

python resnet.py config_lesta_df.ini 
