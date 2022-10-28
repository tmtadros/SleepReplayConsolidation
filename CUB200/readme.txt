This file contains code to run CUB200 experiments in the paper
run_cun_cf_summary.m runs SRA, control NN, and ideal NN experiments

run_sleep_with_replay is a function (called in run_sleep_replay_sweep) to perform training (with/without sleep) with data rehearsal (percent_replay determines amount of data stored)

.mat files contain training and testing data fed through resnet architectures
