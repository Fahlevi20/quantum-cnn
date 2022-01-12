#!/bin/bash
# backup entire experiment folder and then remove tar file
backup_date=date +'%Y%m%d'
tar -zcvf ./experiment_$backup_date.tar.gz ./experiments
cp experiment_$backup_date.tar.gz /mnt/c/Users/mattl/OneDrive/science/projects/qcnn/backup_experiments/
rm experiment_$backup_date.tar.gz
