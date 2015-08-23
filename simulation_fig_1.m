%%
clc; close all;

%--------------------------------------------------------------------------
% This script is an implementation for the first simulation experiment 
% in Supplementary Information (see Sec. 3 and Fig. 1 in Supplementary Information)
%
%
% Network regularized non-negative TRI matrix factorization for PATHway
% identification (NTriPath)
%
% Written by Sunho Park (sunho.park@utsouthwestern.edu)
%            Clinical Sciences department,
%            University of Texas Southwestern Medical Center
%
% This software implements the network regularized non-negative tri-matrix factorization method 
% which integrates somatic mutation data with human protein-protein interaction networks and a pathways database. 
% Driver pathways altered by somatic mutations across different types of cancer
% can be identified by solving the following problem:
%  
%  \min_{S>=0,V>=0} ||X - USV'||_W^2 
%                   + lambda_S ||S||_1^2 + lambda_V||V||_1^2
%                   + lambda_{V0}||V - V0||_F^2 + \lambda_{V_L}tr(V'LV)
%
% where ||X - USV'||_W^2 = sum_i sum_j W_ij(X_ij - [USV]_ij)^2   
%
% Reference:
%      "An integrative somatic mutation analysis to identify pathways 
%          linked with survival outcomes across 19 cancer types"
%      submitted to Nature Communication 
%
% Please send bug reports or questions to Sunho Park.
% This code comes with no guarantee or warranty of any kind.
%
% Last modified 8/22/2015
%--------------------------------------------------------------------------


%--- data generataion (see the mutation matrix X in Fig. 1 in Supplementary Information)
mn_NVBoxes = 5;  % the number of vertical blocks
mn_NHBoxes = 10; % the number of horizontal blocks

mn_NVelinBox = 50;  % the number of samples in a vertical block
mn_NHelinBox = 100; % the number of genes in a horizontal block

M = mn_NVBoxes*mn_NVelinBox; % the nubmer of samples
N = mn_NHBoxes*mn_NHelinBox; % the number of genes

K1 = mn_NVBoxes;
K2 = mn_NHBoxes;

% We considered different sparsity levels of the mutation data matrix 
% for each patient subgroup using probability P
mr_S2 = 0.8;  % P for the block 3
mr_S1 = 0.9;  % P for the rest blocks

mr_S3 = 0.99; % for noise terms

mr_S4 = 0.7; % 

% We generalize the weight loss function in the main paper. We allow W_ij
% to have a nonnegative constant when X_ij is zero.
mr_wX = 0.1; % the weight for the errors on zero entries in X  

% U0: Sample cluster (#samples X #cancer types)
U0 = zeros(M, K1);
for mn_i = 1:K1,            
    m_vidx = (mn_i-1)*mn_NVelinBox + 1:mn_i*mn_NVelinBox; 
    U0(m_vidx, mn_i) = 1;
end

% S0: subgroup-pathway association matrix (#subgroups X #pathways)
S0 = rand(K1, K2);

% V0: Initial pathway information (#pathways X #genes)
mm_Mat = rand(N, K2);
mm_Mat(mm_Mat<mr_S4) = 0;

V0 = zeros(N, K2);
for mn_i = 1:K2,
    m_vidx = (mn_i-1)*mn_NHelinBox + 1:mn_i*mn_NHelinBox; 
    V0(m_vidx, mn_i) = 1;    
end
V0 = V0.*mm_Mat;

% Generate Laplican matrix
% A: Gene-gene interaction network (#genes X #genes types)
% Laplaican matrix L: L = diag(sum(A,2)) - A
mr_SL = 0.90;

A = zeros(N,N);

mm_triuIDX = find(~tril(ones(mn_NHelinBox,mn_NHelinBox)));
for mn_i = 1:mn_NHBoxes,
    mv_idx = (mn_i-1)*mn_NHelinBox + 1:mn_i*mn_NHelinBox;

    mm_randMat = rand(mn_NHelinBox,mn_NHelinBox);        
    mm_randMat(mm_randMat<mr_SL) = 0; 
    mm_randMat = double(mm_randMat>0);
    
    mm_subMat = zeros(mn_NHelinBox,mn_NHelinBox);
    mm_subMat(mm_triuIDX) = mm_randMat(mm_triuIDX);
        
    A(mv_idx,mv_idx) = (mm_subMat+mm_subMat');
end

% D = diag(sum(A,2)) 
D = sum(A, 2); 
    
L = -A;
L(1:N+1:N*N) = D;

% mutation matrix: X
mm_Boxes = zeros(mn_NVBoxes,mn_NHBoxes);

mm_CoulnmSet = cell(5,1);

mm_CoulnmSet{1} = [1,5,6,10];
mm_CoulnmSet{2} = [2,3,5,8];
mm_CoulnmSet{3} = [1:2, 5,7, 8:10];
mm_CoulnmSet{4} = 7;
mm_CoulnmSet{5} = [1,4,9];

for mn_i = 1:5,
    mm_Boxes(mn_i, mm_CoulnmSet{mn_i}) = 1;
end 

mm_Mask = zeros(M, N);
for mn_i = 1:mn_NVBoxes,    
    m_vidx = (mn_i-1)*mn_NVelinBox + 1:mn_i*mn_NVelinBox; 
    mm_Mask(m_vidx,:) = repmat(reshape(repmat(mm_Boxes(mn_i,:)',[1,mn_NHelinBox])',...
                               [N,1])',[mn_NVelinBox,1]);
end

X = rand(M, N).*mm_Mask;

mm_rand = rand(M, N); 
mm_Score = mr_S1*ones(M, N);

mn_i = 3;
mv_idx = (mn_i-1)*mn_NVelinBox + 1:mn_i*mn_NVelinBox; 
mm_Score(mv_idx,:) = mr_S2;

mm_idx = mm_rand < mm_Score;
X(mm_idx) = 0;
X = double(X>0);

[nn, dd] = size(X);
tmp_rand = rand(nn,dd);

tmp_rand(tmp_rand<mr_S3) = 0; 
tmp_rand = double(tmp_rand > 0);

X = double((X + tmp_rand)>0);

figure; hold on;
subplot(2,4,1); imagesc(255-X); colormap(gray); title('X'); 
subplot(2,4,2); imagesc(255-U0); colormap(gray); title('U0'); 
subplot(2,4,3); imagesc(255-V0); colormap(gray); title('V0');
subplot(2,4,4); imagesc(255-A); colormap(gray); title('A');

%--- Nonnegative Matrix Tri-Factorization (NMTF)
% regularization parameters
lamda_V =0.1;
lamda_V0 = 0.1; 
lamda_VL = 0.1;

lamda_S = 0.001;

% the maximum iteration for NtriPath  
Maxiters = 20;
     
fprintf('lamda_V = %3.2f, lamda_{V0} = %3.2f, lamda_{VL} = %3.2f, lamda_S = %3.2f \n',...
         lamda_V, lamda_V0, lamda_VL, lamda_S);
                        
%--- initilaize parameters in the procedure for the inadmissible zeros problem
% kappa: Inadmissible structural zero avoidance adjustment (e.g. 1e-6)
% kappa_tol: Tolerance for identifying a potential structural nonzero (e.g., 1e-10)
% eps: minimum divisor to prevent divide-by-zero (e.g. 1e-10)
kappa = 1e-6;       
kappa_tol = 1e-10;  
eps = 1e-10;

%--- initilaize factor matrices
% U: Patient cluster (#patients X #cancer)
% S: Subgroup-pathway association matrix (#subgroups X #pathways)
% V: Updated pathway information (#pathways X #genes)
% Here, we assume that U is fixed during the learning process. 
U = max(U0, kappa);
S = ones(K1, K2);                     
V = max(V0, kappa);                        

% W (weight matrix): W_ij is 1 if X_ij is non zero, otherwise mr_wX                    
W = X > 0;
W_zero = W == 0;
   
for iter = 1:Maxiters,
    %--- update S
    % X_hat = (U*S*V')oW, where o is an element wise multiplication operator 
    X_hat = (U*S)*V';
    X_hat(W_zero) = mr_wX*X_hat(W_zero);
                            
    % multiplicative factor
    gamma_S = ((U'*X)*V)./( ((U'*X_hat)*V) + lamda_S*sum(sum(S)) + eps);
                            
    % checking inadmissible zeros
    ChkIDX = (S<kappa_tol) & (gamma_S>1);
                            
    S(ChkIDX) = (S(ChkIDX)+kappa).*gamma_S(ChkIDX);
    S(~ChkIDX) = S(~ChkIDX).*gamma_S(~ChkIDX);   
                           
    %- update V                  
    US = U*S;
                            
    X_hat = US*V';
    X_hat(W_zero) = mr_wX*X_hat(W_zero);
              
    % multiplicative factor                            
    gamma_V = ( (X'*US) + lamda_V0*V0 + lamda_VL*(A*V) )./...
                        ( (X_hat'*US) + lamda_V0*V ...
                                      + lamda_VL*(spdiags(D,0,N,N)*V) + lamda_V*sum(sum(V)) + eps );        
    % checking inadmissible zeros
    ChkIDX = (V<kappa_tol) & (gamma_V>1);
                            
    V(ChkIDX) = (V(ChkIDX)+kappa).*gamma_V(ChkIDX);
    V(~ChkIDX) = V(~ChkIDX).*gamma_V(~ChkIDX);
    
    % normalization of V and S
    V_sum = sum(V, 1);
    V = V./repmat(V_sum,[N,1]);
    S = S.*repmat(V_sum,[K1,1]);

    %- display the progress
    if  mod(iter, 10) == 0,
        fprintf('. %5d\n', iter);
    else
        fprintf('.');
    end
end
%-----------------------------------------------------------------%

X_tilde = U*(S*V');
subplot(2,4,5); imagesc(255-X_tilde); colormap(gray); title('X-tilde');
subplot(2,4,6); imagesc(255-U); colormap(gray); title('U (=U0)');
subplot(2,4,7); imagesc(255-S); colormap(gray); title('learned S');
subplot(2,4,8); imagesc(255-V); colormap(gray); title('learned V');
