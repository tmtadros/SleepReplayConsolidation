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

A link to the repository for the neural network library is here:
https://github.com/rasmusbergpalm/DeepLearnToolbox

Modifications made to this library were done in the following repository:
https://github.com/dannyneil/spiking_relu_conversion

To access the CIFAR10 data used in the paper (which were extracted from a VGG net), please see the following Zenodo citation:
Tadros, Timothy. (2022). CIFAR10 Dataset Used in SRC Paper [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7262424


