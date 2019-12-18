#!/bin/bash
#SBATCH -n 1		# Number of tasks
#SBATCH -J ana 	# Name of the job
#SBATCH -p p4		# Partition
#SBATCH -N 1            # number of nodes
#SBATCH -c 1
#SBATCH -o ./out_ana.out
#SBATCH -e ./err_ana.err

python analysis.py config_lesta_df.ini 
