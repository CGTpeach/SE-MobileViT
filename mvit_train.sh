#!/bin/bash
#SBATCH -N 1
#SBATCH -p prod
#SBATCH -t 7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH -J TRAIN
source ~/.bashrc
conda activate class
CUDA_VISIBLE_DEVICES=0 python /public/home/chenguotao2021/classification/MobileViT/train.py
