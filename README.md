# SleepReplayConsolidation
Repository for Nature Communications paper entitled "Sleep-like Unsupervised Replay Reduces Catastrophic Forgetting in Artificial Neural Networks"
This repository contains code needed to run most of the experiments in the paper.

Each dataset folder contains code needed to run the following experiments:
1) Ideal NN (parallel training)
2) Control NN (sequential training)
3) SRA
4) SRA + rehearsal (or rehearsal alone)
5) SRA + iCaRL (or iCaRL alone)
as well as some additional analytical expeirments.

utils/ contains some utility files for each of the tasks

sleep/ contains the main sleep algorithm used in the paper

dlt_cnn_map_dropout_nobiasnn/ contains the neural network training library (which were not modified) and some iCaRL prediction files (which were modified) for the iCaRL NeM classifier
