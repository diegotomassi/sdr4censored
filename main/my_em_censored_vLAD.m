% function [cladMLEparam,cmladMLEparam,ladMLEparam] = my_em_censored_vLAD(x,y, u,xcb,maxIt)
%
% This function applies sufficient dimension reduction for censored data
% using LAD model. Three different estimators are returned:
%   - cLAD
%   - cmLAD
%   - standard LAD
%
%
% INPUTS:
%
% x         N*p     censored observations
% y         N*1     outcome
% u         integer dimension of the reduced subspace
% xcb       2*p     censoring bounds [lower ; upper]
% maxIt     integer maximum number of iterations for the EM procedure
%
%
% OUTPUTS:
%
% Three cell-structures are returned (cladMLEparam,cmladMLEparam,ladMLEparam), 
% each corresponding to one estimator.
% Information inside the cell-structures is the same for the three cases.
%
% For a cathegorical outcome with H different levels, each cell stricture
% has H+1 cells. The first H cells have the ML estimates for model
% parameters in each group. For example:
%    cladMLEparam{h}.mu  <---- MLE of E(X|Y==h)
%    cladMLEparam{h}.sig <---- MLE of COV(X|Y==h)
%
% Cell H+1 in the structure contains overall parameters:
%    cladMLEparam{H+1}.output = basis matrix for the reduction
%    cladMLEparam{H+1}.sigmag = estimate for COV(X)
%    cladMLEparam{H+1}.delta = estimate for E(COV(X|Y))
%    cladMLEparam{H+1}.sigmag = estimate for E(X)
% 
% 
% 
% 
% 
% 
% 
% =========================================================================
function [ladMLEparam,cmMLEparam,MLEparam,globalparam,sampleparam] = my_em_censored_vLAD(x,y, u,xcb,maxIt,init)

if (nargin < 5)||isempty(maxIt),
    100;
end
if nargin<4 || ~exist('xcb', 'var')
    xcb = [-Inf; Inf];
end
if nargin<3,
    error('not enough input arguments');
end


y=grp2idx(y);
nslices=max(y);

% const
[N,d] = size(x);

regVal = eps(max(eig(cov(x))));       % a small number added to diagonal of cov mtx
tol = 1e-10;            % termination tolerance


% cencoring pattern
XX = cell(nslices,1);
xpattern = cell(nslices,1);

% INITIALIZATION
sampleparam = my_em_censored_p(x,y,xcb,5);

for h=1:nslices,
    XX{h} = x(find(y==h),:);
    xpattern{h} = my_em_censor_pattern(XX{h},xcb);
end
% initial reduction
  if (nargin < 7)||isempty(init),  
     sampleparam{nslices+1}.initFlag = logical(0);
     W = aida4censored_v2(sampleparam,u);
     sampleparam{nslices+1}.initvalue = W;
     [beta,fn,fp] = lad4censored_v2(sampleparam,u);
  else 
      beta = init;
      sampleparam{nslices+1}.initFlag = logical(1);
  end
  MLEparam{nslices+1}.initvalue = beta;
  MLEparam = LADsample2mle(sampleparam,beta);
  sampleparam{nslices+1}.maxiter = 1000;
  cmMLEparam = MLEparam;
  
  % ####### initial MLE ##########
  Wo = aida(y,x,u,'disc');
  [~,beta] = ldr(y,x,'LAD','disc',u,'initval',Wo);
  ladMLEparam = MLEparam;
  ladMLEparam{nslices+1}.output = beta;
  ladMLEparam = LADsample2mle(sampleparam,beta);


  %===== save starting value
MLEparam0 = MLEparam;
% 

% ============ EM ===================
ll_old = -Inf;
ll_hist = zeros(maxIt,1);
for it = 1:maxIt
    % compute the log-likelihood
    ll = 0;
    for h=1:nslices,
        ll = ll + my_em_censor_loglik(XX{h},MLEparam{h}.mu,MLEparam{h}.sig,xpattern{h});
    end
    % check convergence
    lldiff = ll - ll_old;
    ll_hist(it) = ll;
    if (lldiff >= 0 && lldiff < tol*abs(ll))
        break;
    end
    ll_old = ll;
    
    % update sample parameters
    mu = zeros(1,d);
    Delta = zeros(d);
    M = zeros(d);
    for h=1:nslices,
        [xhat_e, Q_e, alpha] = myxhatQ(XX{h}, MLEparam{h}.mu, MLEparam{h}.sig, xpattern{h});
        % Numerical issue
        alpha0 = (alpha==0); xhat = xhat_e; xhat(alpha0,:) = 0;Q = Q_e;Q(:,:,alpha0) = 0;
        % UPDATE MEANS
        sampleparam{h}.mu = mean(xhat);
        % UPDATE COVS
        xhatc = bsxfun(@minus,xhat,MLEparam{h}.mu);
        sig0 = (xhatc'*xhatc + squeeze(sum(Q,3)))/sampleparam{h}.n;
        sampleparam{h}.sig = (sig0+sig0')/2;% + regVal*eye(d);
        
        mu = mu + sampleparam{h}.n*sampleparam{h}.mu/N;
        Delta = Delta + sampleparam{h}.n*sampleparam{h}.sig/N;
    end
   
    for h=1:nslices
        M = M + sampleparam{h}.n*(sampleparam{h}.mu -mu)*(sampleparam{h}.mu - mu)'/N;
    end
    Sigma = Delta + M;
    sampleparam{nslices+1}.sigmag = (Sigma+Sigma')/2;% + regVal*eye(d);  
    sampleparam{nslices+1}.delta = (Delta+Delta')/2;% + regVal*eye(d);  
    sampleparam{nslices+1}.mu = mu;
    
    % UPDATE REDUCTION
    if ~sampleparam{nslices+1}.initFlag,
         W = aida4censored_v2(sampleparam,u);
         sampleparam{nslices+1}.initvalue = W;
    end
    [beta,fn,fp] = lad4censored_v2(sampleparam,u);
    MLEparam = LADsample2mle(sampleparam,beta);
    MLEparam{nslices+1}.output = beta;
end


% output
globalparam.iters = it;
globalparam.log_lh = ll;
globalparam.ll_hist = ll_hist(1:it);
globalparam.regVal = regVal;
globalparam.output = MLEparam{nslices+1}.output;
globalparam.initval = MLEparam0{nslices+1}.output;
end
