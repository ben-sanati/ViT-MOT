#!/bin/bash
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bes1g19@soton.ac.uk
python /scratch/bes1g19/UG-Research/ViTDet/experiments.py &> /scratch/bes1g19/UG-Research/ViTDet/MAKE/OUT/experiments_ViTDet.out
