
%% L1-norm Principal-Component Analysis via Bit Flipping
% ---
% Date: September 2016
% ---
% Author:
% Prof. Panos P. Markopoulos, Ph.D.
% Assistant Professor, Department of Electrical and Microelectronic Engineering
% Co-director, Communications Laboratory
% Kate Gleason College of Engineering
% Rochester Institute of Technology 
% Mailing address: 79 Lomb Memorial Drive, Rochester, NY 14623
% Office: Building 09 - Room 3081
% Tel. #: +1 (585) 475-7917
% Fax #: +1 (585) 475-5845
% Email: pxmeee@rit.edu, panos@rit.edu
% Web: http://people.rit.edu/pxmeee/

% ---
% Reference:
% This script approximates the optimal L1-principal components of real-valued data,
% as presented in the article:
% P. P. Markopoulos, S. Kundu, S. Chamadia, and D. A. Pados, 
% ``Efficient L1-norm Principal-Component Analysis via Bit Flipping," 
% submitted to IEEE Transactions on Signal Processing.
% ArXiv: https://arxiv.org/abs/1610.01959
% ---
% Function Description:
% Input: Fat data matrix X,  subspace dimensionality K, number of
% initializations numinit, number of bitflips maxiter, print-results option prnt.
% Output: L1-PCs Q, Binary nuc-norm solution B, L1-PCA metric m, number of
% bitflips nbf per initilization (average)
% ---
% Note:
% Inquiries regarding the script provided below are cordially welcome.
% In case you spot a bug, please let me know.
% If you use some piece of code for your own work, please cite the
% corresponding article above.





function [Q, B, m, a, res, unflpd, ms, ms2, nbf, totaltime, Biters] = l1pca_BF(X, K, init, numinit, maxiter, prnt)

%init: initilization
%numinit: number of initilizations
%maxiter: maximum iterations
%prnt: print

res=0;
unflpd=0;
ms=0;
ms2=0;

[D, N]=size(X);

Biters=zeros(N, K, maxiter);

tic;
bprop=ones(N,1);
maxmetric=-inf;

bfs=zeros(numinit,1);
A=X'*X;
Xt=X;

if K==1
    tic;
    for l = 1:numinit
        
        v=randn(N,1);
        
        if init==1
            v=X'*randn(D,1);
        end 
        if init==1 &&  l<2
            [~, S, V]=svd(X, 'econ');
            Xt=S*V';
            v=V(:,1);
        end
        
        b=sign(v);
        Biters(:,:,1)  = b;
        
        metric=norm(Xt*b)^2;
        a=zeros(1,N);
        for n=1:N
            a(1, n)= 2*b(n)*A(n,:)*b - 2*norm(Xt(:,n))^2;
        end
        ms=metric;
        q=X*b/norm(X*b);
        ms2= sum(abs(X'*q))^2;

        zz=1;
        unflipped=ones(1,N);
        while true
            unflpd(zz,1:N)=unflipped;
            res(zz)=0;
            if prod(unflipped)==1
                res(zz)=1;
            end
            atemp=a(zz,:).*unflpd(zz,:);
            [sa, ind]=sort(atemp,'ascend');
            
            if sa(1)<0  && zz-1<maxiter
                zz=zz+1;
                
                flind=ind(1);
                
                b(flind)=-b(flind);
                Biters(:,:,zz)  =b;
                unflipped(flind)=0;
                a(zz, flind)=-a(zz-1, flind);
                for m=[1:(flind-1),(flind+1):N]
                    a(zz, m)=a(zz-1, m)+4*b(flind)*b(m)*A(m,flind);
                end
            elseif sa(1)>=0 && prod(unflipped)==0 && zz-1<maxiter
                unflipped=ones(1,N);
            else
                bfs(l)=zz-1;
                break;
            end
            
            metric= norm(Xt*b)^2;
            ms(zz)=metric;
            q=X*b/norm(X*b);
            ms2(zz)= sum(abs(X'*q))^2;
        end
        
        if metric>maxmetric
            bprop=b;
            maxmetric=metric;
        end
        for xx=(zz+1):maxiter
            Biters(:,:,xx)=Biters(:,:,zz);
        end
        
    end
    
    B=bprop;
    Q=X*B/norm(X*B);
    m= sum(abs(X'*Q));
    nbf=mean(bfs);
    totaltime=toc;
    
    
else
    
    metric=zeros(maxiter,numinit);
    
    tic;
    for l = 1:numinit
        v=randn(N,1);
        
        if init==1 
            v=X'*randn(D,1);
        end    
        if init==1 &&  l<2
            [~, S, V]=svd(X, 'econ');
            Xt=S*V';
            v=V(:,1);
        end
        
        b=sign(v)*ones(1,K);
        bvec=b(:);
        metric(1,l)=nucnorm(Xt*b);
        a=zeros(1,N*K);
        for nk=1:N*K
            bvectemp=bvec;
            bvectemp(nk)=-bvec(nk);
            btemp=reshape(bvectemp, N ,K);
            a(1, nk)= metric(1,l)-nucnorm(Xt*btemp);
        end
        zz=1;
        unflipped=ones(1,N*K);
        while true
            unflpd(zz,1:N*K)=unflipped;
            atemp=a(zz,:).*unflpd(zz,1:N*K);
            [sa, ind]=sort(atemp,'ascend');
            
            if sa(1)<0  && zz-1<maxiter
                zz=zz+1;
                
                flind=ind(1);
                
                bvec(flind)=-bvec(flind);
                b=reshape(bvec, N ,K);
                unflipped(flind)=0;
                metric(zz,l)= nucnorm(Xt*b);
                for m=1:N*K
                    bvectemp=bvec;
                    bvectemp(m)=-bvec(m);
                    btemp=reshape(bvectemp, N ,K);
                    a(zz, m)= metric(zz,l)-nucnorm(Xt*btemp);
                end
                
                
            elseif sa(1)>=0 && prod(unflipped)==0 && zz-1<maxiter
                unflipped=ones(1,N*K);
            else
                bfs(l)=zz-1;
                break;
            end
                      
            
        end
        
        if metric(zz,l)>maxmetric
            B=b;
            maxmetric=metric(zz,l);
            max_init = l;
        end
    end
    
    [Uf, ~, Vf]=svd(X*B, 'econ');
    
    Q=Uf(:,1:K)*Vf(:,1:K)';
    m= sum(sum(abs(X'*Q)));
    nbf=mean(bfs);
    totaltime=toc;
    
    
end


if prnt
    disp('------------------------------');
    disp(['Avg. #bitflips per initialization: ' num2str(nbf)]);
    disp(['Total time elapsed (sec): ' num2str(totaltime)]);
    disp(['Metric value: ' num2str(m)]);
    disp(['Max Init: ', num2str(max_init)]);
    disp('------------------------------');
end

function nn=nucnorm(A)
[~, S, ~]=svd(A, 'econ');
nn=sum(sum(S));


