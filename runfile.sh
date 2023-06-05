#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=13
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=120:00:00

#Loading modules
module load 2022
module load matplotlib/3.5.2-foss-2022a
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0

#Create output directory on scratch
mkdir "$TMPDIR"/output_dir

# Execute the program 4 times, each program will use 32 cores in parallel. In this example we request all the cores available on the node.
# The '&' sign is used to start each program in the background, so that the programs start running concurrently.

modelversions=(1,3,4,5,6,7,8,9,10,11,12,13,14)

for i in modelversions; do
  $HOME/thesis/main.py $i &
done
wait

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME