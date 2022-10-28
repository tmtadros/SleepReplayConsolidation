This folder contains code to run the following simulations in the paper:
run_class_task_test_summary.m:
1) Ideal NN (parallel training)
2) Control NN (sequential training)
3) SRA
run_sleep_replay_sweep:
4) SRA + small percent replay
run_icarl_sleep:
5) iCaRL alone
6) iCaRL + SRA

CIFAR10 data is stored in 'cifar_data_1000_256_tinyimagenet.mat'
Other relevant files are in ../sleep, ../utils and ../dlt_cnn_map_dropout_nobiasnn
