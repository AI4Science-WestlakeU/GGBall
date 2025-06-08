# GGBall

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n ggball rdkit=2023.03.2 python=3.9```
  - `conda activate ggball`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Install Geoopt: 
    
    ```pip install geoopt```

  - Install pyg:

    ''' pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html'''
  - Run:
    
    ```pip install -e .```

  - Navigate to the ./analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

## Run the code
  
  - All code about HAE or HVQVAE is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py dataset.debug=true`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code with poincare flow matching: `python3 train_flow.py`.
  - You can specify the dataset with `python3 main.py dataset=ego_small`. Look at `configs/dataset` for the list
of datasets that are currently available
    

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: './GGBall/analysis/orca/orca'`: You probably did not compile orca.
    
