# GMDNet
**GMDNet: A Graph-Based Mixture Density Network for Estimating Packages’ Multimodal Travel Time Distribution**
# Usage
Use the following command to run the code:

```
cd algorithm/GMDNet
python train.py
```
# Repo Structure
The structure of our code and description of important files are given as follows:  
│────algorithm/  
│    ├────gmdnet/: code of GMDNet.  
│────data/: a subset of data used for train, validation, and test.  
│────my_utils/  
│    ├────util.py
# Parameters:
| Name          | Type  | Description                                                  |
|:--------------|:------|:-------------------------------------------------------------|
| hidden_dim    | int   | number of hidden units.                                      |
| n_gaussians   | int   | number of Gaussian components.                               |
| att_hidden_size      | int   | number of hidden units in the self-attention.                |
| num_of_attention_heads     | int   | num_of_attention_heads.                                      | 
| num_layers       | int   | number of GNN layers.                                        |
| dirichlet_alpha   | int   | setting of Dirichlet regularizer.                            |
| batch_size     | int   | number of samples per batch.                                 |
| lr  | float | learning rate for training.                                  |
| early_stop      | int   | stop training when a monitored metric has stopped improving. |
| num_epochs    | int   | number of passes over the training data.                     |

