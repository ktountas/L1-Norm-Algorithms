%% Close all
close all;
clear all;
clc;

addpath('../lib')

%% Script Parameters.
D = 6;      % # of rows.
N = 9;      % # of columns.

R = 2;      % # of prinicipal components.

% L1-PCA Parameters Initialization.
l1_init_flag = 1;
l1_init = 1;
l1_niter = 100;
l1_print_flag = 0;

%% Create a DxN real matrix
X = randn(D,N);

%% Perform L1-PCA on the matrix, requesting R principal components.
Q = l1pca_BF(X, R, l1_init_flag, l1_init, l1_niter, l1_print_flag)