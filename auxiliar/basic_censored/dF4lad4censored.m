function df = dF4lad4censored(W,FParameters)
%	Derivative of F (minus the log-likelihood) for the LAD model.
% Inputs:
%    - W: orthogonal basis matrix for the dimension reduction subspace.
%    - FParameters: structure of model parameters and sample statistics.
%
%==========================================================================
K = length(FParameters.param);
sigma = mycell2array(FParameters.param,'sig',K);
sigmag = FParameters.globalparam.sigmag;
nj = mycell2vec(FParameters.param,'n',K); %[FParameters.n{1} FParameters.n{2}];
p = cols(sigmag);
n = sum(nj);
a = zeros(rows(W), cols(W), length(nj));
sigma_i = zeros(p);
    for i=1:length(nj)
        sigma_i = FParameters.param{i}.sig;
        a(:,:,i) = -nj(i)*sigma_i*W*inv(W'*sigma_i*W);
    end
    first = sum(a,3);
    second = n * sigmag * W * inv(W' * sigmag * W);
    % ---Derivative of likelihood function for the LAD model
    df = -(first+second);
