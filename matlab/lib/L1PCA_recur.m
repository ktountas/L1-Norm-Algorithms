%% Recursive L1-norm Principal-Component Analysis
% ---
% Date: November 2019
% ---
% Author:
% Konstantinos Tountas, Ph.D.

% ---
% Reference:
% This script approximates the L1-principal components of real-valued data,
% as presented in the article:
% "Iterative re-weighted L1-norm principal-component analysis"
% Y. Liu, D. A. Pados, S. N. Batalama, and M. J. Medley, in Proc. Asilomar Conference, Pacific Grove, CA, Oct. - Nov. 2017.
% ---
% Function Description:
% Input: Fat data matrix X,  subspace dimensionality K, number of
% iterations maxit, tolerance tol.
% Output: L1-PCs Q, Binary nuc-norm solution B, L1-PCA metric m
% ---
% Note:
% Inquiries regarding the script provided below are cordially welcome.
% In case you spot a bug, please let me know.
% If you use some piece of code for your own work, please cite the
% corresponding article above.

function [Q,B, met]=L1PCA_recur(X,K,maxit,tol)
    metmax=0;
    met=zeros(1,maxit);
    t=0;
    [Q,~,~] = mSVD(X, K);
    while true
       t=t+1;
       A=X'*Q;
       met(t)=sum(abs(A(:)));
       B=sign(A);
      [UK,~,VK]=mSVD(X*B,K);
       Q=UK*VK';
       if (abs(met(t)-metmax)/abs(metmax))<=tol || t==maxit
           break
       else
           metmax=met(t);
       end
    end
    met=met(1:t);
end

function [U,S,V]=mSVD(X,K)
       [UK,SK,VK]=svd(X,'econ');
       S=SK(1:K,1:K);
       U=UK(:,1:K);
       V=VK(:,1:K);
end

