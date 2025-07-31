#  Neighbor-Sampling Based Momentum Stochastic Methods for Training Graph Neural Networks

 These are instructions to reproduce the experiments from [ARXIV LINK TO PAPER]. Our code is a modified version of https://github.com/THUDM/CogDL/tree/master/examples/VRGCN . This document includes:
 - Package requirements for running the experiments
 - How to run the code to reproduce the results from the paper
 - How to modify the code for different datasets or hyperparameter settings


## Package Requirements
The packages required to run these experiments are listed in the requirements.txt file.

## Description of Files
- CV_Adam_Hyperparam_Tuning.py : Runs the hyperparameter tuning experiments and saves the results in .json files. 
- GraphData.py : Loads in datasets, does preprocessing, and stores the fixed hyperparameter values for the five datasets in our paper.
- CV_Adam_Functions.py : Train and test functions.
- VRGCN.py : The control variate model and the historical embedding object from https://arxiv.org/pdf/1710.10568 
- dataloder.py : Creates minibatches with adjacency matrices based on neighbor sampling. 


## Datasets
As written, the code will run for the five benchmark datasets we used in our paper: Cora, CiteSeer, ogbn-arxiv, Flickr, and Reddit. For Cora, CiteSeer, and Reddit, the data is loaded in from torch_geometric.datasets. For ogbn-arxiv, the data is loaded in from ogb.nodeproppred.
To run the code for the Flickr dataset, you will need to have the following files saved in your working directory in a folder named 'flickr': adj_full.npz, feats.npy, class_map.json, role.json . The GraphSAINT GitHub repository https://github.com/GraphSAINT/GraphSAINT?tab=readme-ov-file contains a Google Drive link, where these files can be downloaded.

For more information, please visit the following links:

- Cora and CiteSeer: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html
- ogbn-arxiv: https://ogb.stanford.edu/docs/nodeprop/
- Reddit: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Reddit.html#torch_geometric.datasets.Reddit
- Flickr: https://arxiv.org/pdf/1907.04931

## Running Hyperparameter Tuning Experiments
To run the hyperparameter tuning results for our method, run the CV_Adam_Hyperparam_Tuning.py file. 

The following hyperparameters were held fixed during our experiments:   
|Dataset | Hidden Dimension | Weight Decay | Dropout | Epochs | Runs|
|---|---|---|---|---|---|
|Cora   | 32 |.0005 |.5 | 100| 3|
|CiteSeer   | 32 |.0005 |.5 | 100| 3|
|ogbn-arxiv   | 256 |.00001 |0 | 150| 3|
|Flickr   | 256 |0 |.2 | 200| 3|
|Reddit   | 128 |0 |0 | 30| 1|

The following hyperparameters were tuned from these ranges using grid search:
|Dataset | Learning Rate |Sampled Neighbors | Batch Size |
|---|---|---|---|
|Cora   | {.001,.01,.05}| {2,5}  |{10,20,50}  |
|CiteSeer   |{.001,.01,.05}| {2,5}  |{10,20,50}  |
|ogbn-arxiv   |{.001,.005,.01} | {2,5}  |  {1000,2048,5000} |
|Flickr   |{.1,.5,.8}  | {2,5}  | {1000,2000,5000} |
|Reddit   |{.01,.05} | {2} | {1000} |

For the specified dataset, the model is trained for the specified number of epochs for each set of the hyperparameters, for the specified number of runs. For each dataset, the same initial weights are used for every run 1, then a different set of initial weights for     run 2, etc. The results are saved in a .json files. There is one .json file for each optimizer for each dataset, with the name "dataset_optimizer_hyperparam_experiments.json" . Each of these .json files saves a list of lists for all runs of training accuracy, test accuracy, validation accuracy and loss.

Since there is randomness in the batch selection and the neighbor sampling, running this code will produce results that are close to the results reported in the paper, but not exactly the same. 

## Modifications for Other Problem Settings
To run the code for a different dataset, the following variables need to be set manually (rather than extracting them as attributes from the GraphData object):
-num_in_channels
-num_out_channels
-epochs
-weight_decay
-dropout
-hidden_size
The data object that is an output from GraphData.data_preprocessing() will also need to be set manually. This is a CogDL Graph() object, with values set for the edge_index, x, y, train_mask, test_mask, and val_mask parameters.
For more information on CogDL graphs, please visit the following documentation: https://docs.cogdl.ai/en/latest/tutorial/graph.html
The following code is used to set the full adjacency matrix weights and ensure that the adjacency matrix has self connections:

```
#adding self_loops 
data.add_remaining_self_loops()
#symmetric normalization of the adjacency matrix
degrees=data.degrees() #diag of D matrix
degrees_inv_sqrt=degrees.pow(-0.5)
degrees_inv_sqrt[degrees_inv_sqrt==float("inf")]=0
edge_weights=data.edge_weight
row,col=data.edge_index
data.edge_weight=degrees_inv_sqrt[col]*edge_weights*degrees_inv_sqrt[row]
```

  
    
