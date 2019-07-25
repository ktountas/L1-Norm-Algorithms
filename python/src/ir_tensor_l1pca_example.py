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
# This script serves as an example on how to use ir_tensor_l1pca_v0.py to calculate the L1-norm principal
# components and the data conformity of a random tensor.
#
# Outputs: Q => L1-PCs, 
#		   W => Data conformity tensor.
# ---
# Note:
# Inquiries regarding the script provided below are cordially welcome.
# In case you spot a bug, please let me know.
# If you use some piece of code for your own work, please cite the
# corresponding article above.
# 
#----------------------------------------------------------------------#
import sys
sys.path.insert(1,'../lib')	
from ir_tensor_l1pca_v0 import *

def main():
	# Parameters:
	rank_r = [2, 2, 2]	# Number of L1-norm principal components.
	num_init = 10 		# Number of L1-norm PCA initializations.
	print_flag = False	# Print L1-norm PCA statistics (True/False).
	n_max = 5			# Number of L1-norm Tensor iterations.
	
	# Tensor dimensions.
	D = 3				# Tensor first dimension (rows).
	L = 4				# Tensor second dimension (columns).
	N = 5				# Tensor third dimension.

	# Create a random tensor of dimension DxLxN.
	X = np.random.randn(D, L, N)

	# The weights corresponding to each dimension.
	A = [1/3, 1/3, 1/3]
	
	# Call the L1-norm PCA function.
	Q, W = ir_tensor_l1pca(X, rank_r, A, n_max, num_init, print_flag)

	# Print the calculated subspace matrices.
	print('Calculated subspaces:')
	for ii in range(0, len(X.shape)):
		print(Q[ii])

	# Print the calculated conformity values.
	print('Calculated confomrmity values:')
	for ii in range(0, len(W.shape)):
		print(W[ii]) 

if __name__ == '__main__':
	try:
		main()
	except Keyboardfloaterrupt:
		pass
