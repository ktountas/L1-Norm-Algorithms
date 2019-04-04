from l1pca_sbfk_v0 import *

def main():
	# Parameters:
	rank_r = 2	    	# Rank of the desired subspace.
	num_init = 10 		# Number of initializations.
	print_flag = True	# Print True/False
	X = np.random.randn(9, 6)
	
	# Call the L1-norm PCA function.
	Q, B = l1pca_sbfk(X, rank_r, num_init, print_flag)

if __name__ == '__main__':
	try:
		main()
	except Keyboardfloaterrupt:
		pass