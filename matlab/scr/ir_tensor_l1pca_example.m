%% Close all
close all;
clear all;
clc;

addpath('../lib')

%% Script Parameters.
D = 3;     % # of rows.
L = 4;      % # of columns.
N = 5;     % # of samples.

% L1-PCA Parameters Initialization.
R_1 = 3;    % # of prinicipal components corresponding to the first dimension.
R_2 = 2;    % # of prinicipal components corresponding to the second dimension.
R_3 = 4;    % # of prinicipal components corresponding to the third dimension.
ranks_prop = [R_1, R_2, R_3];

weights = [1/3, 1/3, 1/3];  % Weights corresponding to the importance of each dimension.

l1_niter = 10;
l1_print_flag = 0;

%% Create a DxLxN real tensor.
X = randn(D, L, N);

%% Perform L1-PCA on the matrix, requesting R principal components.
[Q_prop, W] = ir_tensor_l1pca_stable(X, l1_niter, ranks_prop, weights, l1_print_flag);

Q_1 = Q_prop{1}
Q_2 = Q_prop{2}
Q_3 = Q_prop{3}

W.data