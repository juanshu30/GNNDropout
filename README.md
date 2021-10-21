# Understanding Dropout for Graph Neural Network

About
====

This directory contains the code and resources of the following paper:

"Understanding Dropout for Graph Neural Network". Under review.

- We study the different dropout schemes in graph neural network, which has been used widely used but lack of careful study and deep understanding. We first study the difference between dropout in GNNs and CNNs and then propose two novel dropout schemes that can overcome the current drawbacks of GNN dropout schemes. We then carefully study the performance of different dropout schemes in different settings. Lastly, we theorettically study how different dropout schemes can help mitigate over-smoothing problems.


Overview of the Model
====
The model diagram can be viewed from the figure.
<img width="400" height="600" src="https://github.com/juanshu30/GNNDropout/blob/main/Figure/LCD_AHD.pdf"/>


Sub-directories
====
- #[figures] contains the figures that are used in the paper.
- #[shallow Network] contains implementation of our model for the disease gene network.
- ##PPI
- ##cora
- #[Deep Network] contains implementation of our model for the disease gene network.
- ##PPI
- ##cora

Data
====
There are two benchmark datasets used in this study and both of them can be loaded from pytorch geometric

### Cora (Evaluate Transductive performance)
- 2708 nodes and 10556 edges.
- 1433 node features and without edge features
- train_val_test split: 1208\500\1000
- multi-class classification

### PPI (Evaluate Inductive performance)
- 44,906 nodes and 1,226,368 edges
- 50 node features and without edge features
- train_val_test split: 20\2\2(graphs)
- multi-label classification


Code Usage
====
- To run the code, you need python (3.7 I used) installed and other packages, such as pytorch(1.5.0), pytorch-geometric(1.6.1), numpy, pandas, matplotlib. 
- 
