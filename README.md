# CN-SDE

waitting change !!!!

This is a pytorch implementation of CN-SDE, a causal neural network of stochastic differential equation.

![](https://github.com/Starsm7/CN-SDE/blob/main/visual/imgs/CN-SDE.png)

The schematic illustration of the MuiltHipPoseNet algorithm. First, we model the cellular dynamics as the forward process of a diffusion model, whereby the drift term of the SDE learns by discovering causal relationships between genes. Next, considering the consistency loss of cell type distribution between the predictions and the real data, we introduce an additional pre-trained classifier, ROCKET, to classify the cell types of unseen points. Then, through the reverse process of the diffusion model, the gradient term is separated from the drift term of the reverse process, thereby obtaining the Waddington potential landscape of the cells. Finally, jointly train the forward process and reverse process of the diffusion model to ensure reversible consistency between the forward and reverse processes.

## Introduction

We have provided the code for implementing causal networks and causal consistency, bidirectional SDEs, as well as the simulated dataset, respectively.

## Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests/feedbacks.
