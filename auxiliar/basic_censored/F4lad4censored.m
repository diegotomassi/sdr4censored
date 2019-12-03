function f = F4ladcensored(W,FParameters)
%
% f = F4lad(W,FParameters)
%
% This function computes the negative of the log-likelihood for the LAD
% model. 
%
% Inputs:
%    - W: orthogonal basis matrix for the dimension reduction subspace.
%    - FParameters: structure of parameters computed from the sample. It
%    contains:
%          - FParameters.sigma = array of conditional covariance matrices
%          - FParameters.sigmag = marginal covariance matrix
%          - FParameters.n: sample size for each value of teh response Y.
%
%
%==========================================================================
K=length(FParameters.param);
sigma = mycell2array(FParameters.param,'sig',K);
sigmag = FParameters.globalparam.sigmag;
nj = mycell2vec(FParameters.param,'n',K); %[FParameters.n{1} FParameters.n{2}];
n = sum(nj);
p = cols(sigmag);
% ---define some convenience variables
h = nj/n;

a = zeros(length(h),1);
for i=1:length(h),
    CC = tospd(W'*FParameters.param{i}.sig*W);
%     isposdef(CC);
    a(i) = logdet(CC);
end
% ---Likelihood function for LAD model
f = -1*n/2 * (logdet(W'*sigmag*W) - h'*a);
