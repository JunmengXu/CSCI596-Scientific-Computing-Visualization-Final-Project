#!/bin/bash

input_filename="capybara.jpg"

M_values=(4 4 16)
N_values=(4 16 16)
ntasks_values=(1 2 4) 
cpus_per_task_values=(2 4) 

g++ -o evaluation compression_evaluation.cpp -fopenmp -lm -std=c++11

for ntasks in ${ntasks_values[@]}; do
    for cpus_per_task in ${cpus_per_task_values[@]}; do
        for i in ${!M_values[@]}; do
            M=${M_values[$i]}
            N=${N_values[$i]}

            output_filename="capybara_${M}_${N}_${ntasks}_${cpus_per_task}.jpg"

            job_script="capybara_job_${ntasks}_${cpus_per_task}_${M}_${N}.sh"
            cat <<- EOF > $job_script
#!/bin/bash
#SBATCH --job-name=batch_image_compression
#SBATCH --nodes=1
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --time=10:00:00
#SBATCH --output=batch_compression_%j.log

./evaluation ./$input_filename ./$output_filename $M $N 1 $ntasks $cpus_per_task
EOF

            sbatch $job_script
        done
    done
done
