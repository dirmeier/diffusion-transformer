#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=sd31
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/snx3000/sdirmeie/PROJECTS/diffusion-transformer/experiments/mnist_sdf/workdir/slurm/%j.out
#SBATCH --error=/scratch/snx3000/sdirmeie/PROJECTS/diffusion-transformer/experiments/mnist_sdf/workdir/slurm/%j.out

module load daint-gpu
conda activate cifma-uqma-dev

srun python main.py \
  --config=config.py \
  --workdir=/scratch/snx3000/sdirmeie/PROJECTS/diffusion-transformer/experiments/mnist_sdf/workdir/ \
  --usewand
