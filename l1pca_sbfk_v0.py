#! /usr/bin/python
#----------------------------------------------------------------------#
# 
# Date: April 2018
#	
# Author: Konstantinos Tountas
# Research Assistant and Ph.D. Candidate, Dep. of Electrical and Computer Eng. & Computer Science
# Florida Atlantic University
# Mailing address: 777 Glades Rd., Boca Raton, FL, 33431
# Email: ktountas2017@fau.edu
# Web: https://ktountas.github.io/
#
# Dependencies:
# 
#----------------------------------------------------------------------#

import numpy as np
import time
from scipy.linalg import svd

def l1pca_sbfk(X, K, L, print_flag):
	# Parameters
	toler =10e-8;

	# Get the dimentions of the matrix.
	dataset_matrix_size = X.shape	
	D = dataset_matrix_size[0]	# Row dimension.
	N = dataset_matrix_size[1]	# Column dimension.

	# Initialize the matrix with the SVD.
	U_x, S_x, V_x = svd(X , full_matrices = False)	# Hfloat: The singular values are in vector form.
	X_t = np.matmul(np.diag(S_x),V_x.transpose())

	# Initialize the required matrices and vectors.
	Bprop = np.ones((N,K),dtype=float)
	nucnormmax = 0
	iterations = np.zeros((1,L),dtype=float)

	# For each initialization do.
	for ll in range(0, L):
		
		start_time = time.time()	# Start measuring execution time.

		v = np.random.randn(N,K)	# Random initialized vector.
		if ll<2:	# In the first initialization, initialize the B matrix to sign of the product of the first 
					# right singular vector of the input matrix with an all-ones matrix.
			z = np.zeros((N,1),dtype=float)
			z = V_x[:,0]
			z_x = z.reshape(N,1)
			v = np.matmul(z_x,np.ones((1,K), dtype=float))
		B = np.sign(v)	# Get a binary vector containing the signs of the elements of v.

		# Calculate the nuclear norm of X*B.
		X_temp = np.matmul(X_t,B)
		dummy1, S, dummy2 = svd(X_temp , full_matrices = False)
		nucnorm = np.sum(np.sum(np.diag(S)))
		nuckprev = nucnorm*np.ones((K,1), dtype=float)

		# While not converged bit flip.
		iter_ = 0
		while True:
			iter_ = iter_ + 1

			flag = False

			# Calculate all the possible binary vectors and all posible bit flips.
			for k in range(0, K):

				a = np.zeros((N,1), dtype=float)

				for n in range(0, N):
					B_t = B
					B_t[n,k] = -B[n,k]
					dummy1, S, dummy2 = svd(np.matmul(X_t,B), full_matrices=False)
					a[n] = sum(sum(np.diag(S)))
				
				ma = np.max(a)	# Find which binary vector and bit flips maximize the quadratic.
				if ma > nucnorm:
					nc = np.where(a == ma)
					B_t[nc[0],k] = -B_t[nc[0],k]
					nucnorm = ma

				# If the maximum quadratic is attained, stop iterating.
				if iter_ > 1 and nucnorm<nuckprev[k] + toler:
					flag = True
					break

				nuckprev[k] = nucnorm # Save the calculated nuclear norm of the current initialization.

			if flag == True:
				break

		# Find the maximum nuclear norm across all initializations.
		iterations[0,ll] = iter_
		if nucnorm > nucnormmax:
			nucnormmax = nucnorm
			Bprop = B

	# Calculate the final subspace.
	U, dummy, V = svd(np.matmul(X,Bprop), full_matrices=False)
	Uprop = U[:,0:K]
	Vprop = V[:,0:K]
	Qprop = np.matmul(Uprop,Vprop.transpose())

	end_time = time.time()	# End of execution timestamp.
	timelapse = (end_time - start_time)	# Calculate the time elapsed.

	convergence_iter = np.mean(iterations, dtype=float) # Calculate the mean iterations per initialization.
	vmax = sum(sum(abs(np.matmul(Qprop.transpose(),X))))
	
	# If print true, print execution statistics.
	if print_flag:
		print "--------------------------------------"
		print "Avg. iterations/initialization: ", (convergence_iter)
		print "Time elapsed (sec): ", (timelapse)
		print "Metric value:", vmax
		print "--------------------------------------"

	return Qprop, Bprop