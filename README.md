# CN-SDE

This is a pytorch implementation of CN-SDE, a causal neural network of stochastic differential equation.

![](https://github.com/Starsm7/CN-SDE/blob/main/visual/imgs/CN-SDE.png)

The schematic illustration of the MuiltHipPoseNet algorithm. Firstly, it models cellular differentiation as a stochastic process, as described by the SDE, whereby the drift term of the SDE learns by discovering causal relationships between genes. Next, considering the consistency of cell type distribution between the predictions and the real data, we introduce an additional pre-trained classifier, ROCKET, to classify the cell types of unseen points. Finally, based on the pairwise pairing between the observed distributions, we used conditional flow matching to generate the Waddington potential energy landscape of cells, and learning them through neural networks extends the framework of DeepROUT.

We benchmark the performance of CN-SDE on two time-series scRNA-seq datasets, including pancreatic $\beta$-cell differen­tiation, mouse hematopoiesis datasets, and compare the performance of CN-SDE with cur­rent state-of-the-art time-series scRNA-seq inference methods, namely TrajectoryNet, PRESCIENT, MIOFlow and PI-SDE.

## Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests/feedbacks.
