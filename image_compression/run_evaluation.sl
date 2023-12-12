#!/bin/bash

#SBATCH --job-name=batch_image_compression    # Job name
#SBATCH --nodes=1                              # Run all processes on a single node
#SBATCH --ntasks=1                             # Number of tasks (processes)
#SBATCH --cpus-per-task=1                      # CPU cores per task
#SBATCH --time=02:00:00                        # Time limit hrs:min:sec
#SBATCH --output=batch_compression_%j.log      # Standard output and error log

# Load any modules needed for your application
# module load gcc/9.2.0  (for example)

# Compile program
g++ -o evaluation image.cpp compression_evaluation.cpp -std=c++11 # Replace with your actual source files and flags
# Define sets of parameters or files
input_files=("input1.jpg" "input2.jpg" "input3.jpg")  # Example input files
output_files=("output1.jpg" "output2.jpg" "output3.jpg")  # Corresponding output files
M_values=(4 16 64)  # Example M values
N_values=(4 16 64)     # Example N values

# SLURM job parameters
nodes=$SLURM_JOB_NUM_NODES
ntasks=$SLURM_NTASKS
cpus_per_task=$SLURM_CPUS_ON_NODE

# Loop over sets of parameters or files
for i in ${!input_files[@]}; do
    input_filename=${input_files[$i]}
    output_filename=${output_files[$i]}
    M=${M_values[$i]}
    N=${N_values[$i]}

    # Run your program with each set of parameters
    ./your_program $input_filename $output_filename $M $N $nodes $ntasks $cpus_per_task
done
