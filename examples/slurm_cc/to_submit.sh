#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=0:05:00
#SBATCH --job-name=yoyotest
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log
# remove one # if you prefer receiving emails
##SBATCH --mail-type=all
##SBATCH --mail-user=yoyo@dodo.co

export MLFLOW_TRACKING_URI='mlruns'

main --data ../data --output output --config config.yaml --disable-progressbar
