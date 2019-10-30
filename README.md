# L1-Norm-Algorithms
The current repository provides the code for the popular L1-norm Pricipal Component Analysis for matrix and tensor data sets.

The matrix algorithm was developed and published by P. P. Markopoulos, S. Kundu, S. Chamadia and D. A. Pados, ["Efficient L1-Norm Principal-Component Analysis via Bit Flipping"](https://ieeexplore.ieee.org/document/7934025), in IEEE Transactions on Signal Processing, vol. 65, no. 16, pp. 4252-4264, Aug. 2017.

The tensor algorithm was developed and published by K. Tountas, D. A. Pados, M. J. Medley, ["Conformity Evaluation and L1-norm Principal-Component Analysis of Tensor Data"](https://ktountas.github.io/pdf/spie_2019.pdf), in SPIE Big Data: Learning, Analytics, and Applications Conf., SPIE Defence and Commercial Sensing, Baltimore, MD, Mar. 2019.

## Python Implementation:

The entry point for the matrix algorithm is the file l1_pca_example.py.
The entry point for the tensor algorithm is the file ir_tensor_l1pca_example.py.

We have tested the code on Python 3.7.*. The prerequisite packages to run it are: 
- scipy (publicly available from: https://www.scipy.org/install.html)
- tensorly (publicly available from: http://tensorly.org/stable/installation.html)

The prerequisite packages can be installed via pip: 
pip install -r requirements.txt

List of files included: l1_pca_example.py, l1pca_sbfk_v0.py, ir_tensor_l1pca_v0.py, ir_tensor_l1pca_example.py

## MATLAB Implementation:

The entry point for the matrix algorithm is the file l1_pca_example.m.
The entry point for the tensor algorithm is the file ir_tensor_l1pca_example.m.

We have tested the code on MatlabR2019a. The prerequisite packages to run it are:
- Tensor Toolbox Version 2.6 (can be downloaded from: http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html)
- L1-PCA Toolbox (can be downloaded from: https://www.mathworks.com/matlabcentral/fileexchange/64855-l1-pca-toolbox)

List of files included: l1_pca_example.m, l1pca_BF.m, ir_tensor_l1pca_stable.m, ir_tensor_l1pca_example.m
