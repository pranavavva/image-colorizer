#!/bin/bash
#SBATCH --job-name=imagenet-colorizer         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1         # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=avva@cs.princeton.edu

module purge
source venv/bin/activate
cd image-colorizer
python model2.py