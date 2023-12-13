#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:59
#SBATCH --output=pro.out
#SBATCH -Aanakano_429

./image_mpi ./capybara.jpg ./output.jpg 2 8