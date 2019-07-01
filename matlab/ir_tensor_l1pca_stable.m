function [Q_i, final_weight_tensor] = l1_tensor_normalization_v6(X, ...
    n_iter_max, irw_comp, alpha, parallel_flag)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% L1-PCA Parameters Initialization.
l1_init_flag = 1;
l1_init = 1;
l1_niter = 100;
l1_print_flag = 0;

% If the data tensor is complex, realify.
if ~isreal(X)
    X_realified = realify(X);
else
    X_realified = X;
end

% Use tensor toolbox to create the tensor.
X_ten = tensor(X_realified);
I = X_ten.size;
dims_ = ndims(X_ten);

% Initialize the required cell variables.
Q_i = cell(1,dims_);
mode_i = cell(1,dims_);
init_temp = cell(1,dims_);

% Initialization step: for each dimension create the unfolding and ... 
% calculate the initial subspace.
for ii = 1:length(alpha)
    temp_mode = tenmat(X_ten,ii);   % Crete the unfolding.
    init_temp{ii} = temp_mode;      % Store the tenmat object.
    mode_i{ii} = temp_mode.data;    % Get the unfolding matrix.
    
    % Calculate the L1-PCA subspace.
    Q_i{ii} = l1pca_BF(cell2mat(mode_i(ii)), irw_comp(ii), l1_init_flag, ...
        l1_init, l1_niter, l1_print_flag);
end

% Iterative calculation of weighted L1-norm PCs.
for i_iter = 1:n_iter_max
    
    final_weight_tensor = tensor(zeros(I)); % Initialize the weight tensor.
    
    for ii = 1:length(alpha)
        % Calculate the conformity values of the corresponding dimension.
        weight_tilde_i = vecnorm((cell2mat(Q_i(ii))*((cell2mat(Q_i(ii))).'))...
        *cell2mat(mode_i(ii)));
        
        temp_tenmat = init_temp{ii};    % Initialize local tenmat object.
        temp_tenmat(1:end,1:end) = repmat(weight_tilde_i,I(ii),1);  % Update tenmat object.
        weight_tilde_i_tensor  = tensor(temp_tenmat);   % Conver to tensor object.
        
        % Weight the weight tensor.
        final_weight_tensor = final_weight_tensor + weight_tilde_i_tensor.*alpha(ii);
    end
    
    % Normalize values in [0,1] range.
    normA = final_weight_tensor - min(final_weight_tensor.data(:));
    final_weight_tensor = normA./max(normA(:));
    
    % Weigh the tensor with the corresponding conformity values.
    X_temp = X_ten.*final_weight_tensor;
    
    for ii = 1:length(alpha)
        temp_mode = tenmat(X_temp,ii);    % Create tenmat object of the unfolding.
        
        % Calculate the L1-norm subspaces.
        Q_i{ii} = l1pca_BF(temp_mode.data, irw_comp(ii), l1_init_flag, ... 
            l1_init, l1_niter, l1_print_flag);
    end
end
end