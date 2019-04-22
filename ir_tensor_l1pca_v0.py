#! /usr/bin/python
#----------------------------------------------------------------------#
# ---
# Date: April 2018
# ---
#	
# Author: Konstantinos Tountas
# Research Assistant and Ph.D. Candidate, Dep. of Electrical and Computer Eng. & Computer Science
# Florida Atlantic University
# Mailing address: 777 Glades Rd., Boca Raton, FL, 33431
# Email: ktountas2017@fau.edu
# Web: https://ktountas.github.io/
#
# ---
# Reference:
# This script calculates the L1-norm tensor principal components of real-valued tensor data sets,
# as presented in the article:
# K. Tountas, D. A. Pados, M. J. Medley "Conformity Evaluation and L1-norm Principal-Component 
# Analysis of Tensor Data" 
# in SPIE Big Data: Learning, Analytics, and Applications Conf., SPIE Defence and Commercial Sensing, 2019.
#
# ---
# Function Description:
# Inputs: X => Real valued tensor,
#		  K => Vector containing the subspace dimensionality for each dimension,
#		  A => Vector containing the weights of each dimension.
#		  n_max => Maximum numver of iterations		
#		  num_init => Number of initializations of the L1-norm PCA,
#	      print_flag => Print statistics option.
# Outputs: Q => L1-PCs, 
#		   W => Data conformity tensor.
# 
# ---
# Dependencies:
#	1) PYTHON 3 only. 
#	2) tensorly (http://tensorly.org/stable/installation.html)
#
# ---
# Note:
# Inquiries regarding the script provided below are cordially welcome.
# In case you spot a bug, please let me know.
# If you use some piece of code for your own work, please cite the
# corresponding article above.
# 
#----------------------------------------------------------------------#
from l1pca_sbfk_v0 import *
from tensorly import unfold, fold
from numpy import matlib

def ir_tensor_l1pca(X, K, A, n_max, num_init, print_flag):

	dataset_matrix_size = X.shape
	dataset_matrix_size = list(dataset_matrix_size)
	
	# Initialize the unfolding, subspace, and conformity lists.
	unfolding_ii = [[] for xx in range(len(dataset_matrix_size))]	# Unfolding list.
	Q_ii = [[] for xx in range(len(dataset_matrix_size))]			# Subspace list.
	conf_ii = [[] for xx in range(len(dataset_matrix_size))]		# Conformity list.

	# Calculate the initial subspaces.
	for ii in range (0, len(dataset_matrix_size)):
		unfolding_ii[ii] = unfold(X,ii)		# Calculate the unfoldings.
		
		Q_ii[ii], B, vmax= l1pca_sbfk(unfolding_ii[ii], K[ii], num_init, print_flag)	# Calculate the subspaces.
	
	# Iterate.
	for iter_ in range(0, n_max):

		for ii in range(0, len(A)):
			# Calculate the norm of the projection of each column of the unfolding.
			vect_weight = np.linalg.norm(np.matmul(np.matmul(Q_ii[ii],Q_ii[ii].transpose()),unfolding_ii[ii]), axis = 0)
			# Convert to tensor form and multiply with the corresponding weight
			conf_ii[ii] = np.array(fold(np.matlib.repmat(vect_weight, dataset_matrix_size[ii],1), ii, X.shape), dtype=float)*A[ii]
		
		# Combine the conformity values to form the final conformity tensor.	
		conf_final = zero_one_normalization(sum(conf_ii))

		# Calculate the updated L1-PCs.
		for ii in range(0, len(A)):
			Q_ii[ii], B, vmax= l1pca_sbfk(unfold(np.multiply(X,conf_final),ii), K[ii], num_init, print_flag)

	return Q_ii, conf_final

def zero_one_normalization(X):
	return (X - min(X.flatten()))/(max(X.flatten()) - min(X.flatten()))
