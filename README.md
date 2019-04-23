# L1-Norm-Algorithms
The current repository provides the code for the popular L1-norm Priciplal Component Analysis for matrix and tensor data sets.

The original matrix algorithm was developed and published by P. P. Markopoulos, S. Kundu, S. Chamadia and D. A. Pados, "Efficient L1-Norm Principal-Component Analysis via Bit Flipping", in IEEE Transactions on Signal Processing, vol. 65, no. 16, pp. 4252-4264, 15 Aug.15, 2017.

The original tensor algorithm was developed and published by K. Tountas, D. A. Pados, M. J. Medley, "Conformity Evaluation and L1-norm Principal-Component Analysis of Tensor Data" in SPIE Big Data: Learning, Analytics, and Applications Conf., SPIE Defence and Commercial Sensing, Baltimore, MD, 2019.

The entry point for the matrix case is the file l1_pca_example.py.
The entry point for the tensor case is the file ir_tensor_l1pca_example.py.

We have tested the code on Python 3.7.* The prerequisite packages to run it are: 
- scipy (publicly available from: https://www.scipy.org/install.html)
- tensorly (publicly available from: http://tensorly.org/stable/installation.html)

The prerequisite packages can be installed via pip by: 
pip install -r requirements.txt

List of files included: l1_pca_example.py, l1pca_sbfk_v0.py, ir_tensor_l1pca_v0.py, ir_tensor_l1pca_example.py
