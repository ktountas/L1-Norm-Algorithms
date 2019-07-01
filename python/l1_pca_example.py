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
# This script approximates the optimal L1-principal components of real-valued data,
# as presented in the article:
# P. P. Markopoulos, S. Kundu, S. Chamadia, and D. A. Pados, 
# ``Efficient L1-norm Principal-Component Analysis via Bit Flipping" 
# in IEEE Transactions on Signal Processing, vol. 65, no. 16, pp. 4252-4264, 15 Aug.15, 2017.
#
# ---
# Function Description:
# This script serves as an example on how to use l1pca_sbfk_v0.py to calculate the L1-norm principal
# components of a random matrix.
#
# Outputs: Q => L1-PCs, 
#		   B => Binary nuc-norm solution,
#		   vmax => L1-norm PCA value.
# ---
# Note:
# Inquiries regarding the script provided below are cordially welcome.
# In case you spot a bug, please let me know.
# If you use some piece of code for your own work, please cite the
# corresponding article above.
# 
#----------------------------------------------------------------------#
from l1pca_sbfk_v0 import *
from array import *

def main():
	# Parameters:
	rank_r = 2	    	# Number of L1-norm principal components.
	num_init = 10 		# Number of initializations.
	print_flag = True	# Print statistics True/False.
	
	D = 6				# Matrix row dimension.
	N = 9				# Matrix column dimension

	# Create a random matrix of dimension DxN.
	X = np.random.randn(D, N)
	
	# Call the L1-norm PCA function.
	Q, B, vmax= l1pca_sbfk(X, rank_r, num_init, print_flag)

	print(Q) # Print the calculated subspace matrix.

if __name__ == '__main__':
	try:
		main()
	except Keyboardfloaterrupt:
		pass
