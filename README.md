# Code, Jupyter Notebooks and figures for a thesis/paper (working title)

This repository contains the code used to produce results in a working project in the Pillow Lab that is broadly working to expand/refine existing recurrent network models for context-dependent computations in PFC and the neural codes associated with these computations. This readme outlines the contents of the repository and how they work together, going by directory.

## code/
This contains all the raw `.py` files used in the project. All the files are commented with descriptions for each function/class; this description serves only as a schematic for how to use all the components together. The files can be effectively divided into three categories:

1. Experiments
  - `CleanNetwork.py` - A simple RNN network object
  - `CleanDriver_ff.py` - Provides functionality for training a network with full-FORCE and testing it
  - `CleanDriver_v1.py` - Provides functionality for training a network with FORCE and testing it (model2)
  - `CleanDriver_v2.py` - Provides functionality for training a network with FORCE and testing it (model3)
  - `CleanDriver_vpca.py` - Provides functionality for training a network with FORCE and testing it (pca_model)
2. Fixed point analysis
  - `dynamicalAnalysis.py` - Provides several functions for finding fixed points within RNN state space
3. Neural data handlers
  - `pfc.py` - Processing, smoothing neural data from Mante et. al. 2013
  - `pfcaux.py` - Auxiliary functions for `pfc.py`

## notebooks/
This folder contains some Jupyter Notebooks for training/testing a network and the code used to plot the figures that are in the figures folder. Reading through these notebooks can provide an understanding of the workflow for the code files above, but we provide a basic structure here.

`CleanDriver*.py` provides a Driver object which instantiates a network from `CleanNetwork.py` with paramters passed to a Driver object by the user. The Driver object then provides two functions: `Driver.train(target, context, ...)` and `Driver.test(target, context, ...)`. The train function trains the network to output target when presented with information contained in context. The file name (`_ff`, `_v1`, etc.) dictates which training method is used and what model is used (for
example, what the readout of the network is, mathematically). `Driver.test()` returns the output of the network as a 2d-array to make plotting easy. Testing only requires the target function so that it knows for how long to run the simulation. 

Once a network is trained, various functions in `dynamicalAnalysis.py` provide the ability to locate and analyze fixed points in the state space of the RNN. Inspect the file for descriptions of each function; the notebooks should be good guidance for how to use that tool.

## figures/
Pretty self-explanatory. The files are organized such that both the illustrator and png versions of a figure are available. The figures are divided into categories that relate them to schematizing a model, presenting results, and what stage in the drafting process they are in.


