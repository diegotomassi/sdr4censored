function [sampleparam,param] = my_em_censored_p(x,y,xcb,maxIt)
% em algorithm for censored data (as proposed by Lee&Scott)
% 
% x         N*d     censored observations
% y         N*1     response
% xcb       2*d     censoring bounds [lower ; upper]
% maxIt     Integer maximum number of iterations
% ======================================================
if (nargin < 4)||isempty(maxIt),
    %maxIt = 10*size(x,2);
    maxIt = 5;
end
if nargin<2,
    error('not enough input arguments');
end

if nargin<3 || ~exist('xcb', 'var')
    xcb = [-Inf; Inf];
end


% const
[N,d] = size(x);
regVal = eps(max(eig(cov(x))));        % a small number added to diagonal of cov mtx
% regVal = 1e-15;
tol = 1e-10;            % termination tolerance


% cencoring pattern
y = grp2idx(y);
H = max(y);
XX = cell(H,1);
xpattern = cell(H,1);
loglik = 0;
for h=1:H,
    XX{h} = x(find(y==h),:);
    xpattern{h}= my_em_censor_pattern(XX{h}, xcb); 
    sampleparam{h}.mu = mean(XX{h});
    sampleparam{h}.sig = get_cov(XX{h}) + regVal*eye(d);
    sampleparam{h}.n = size(XX{h},1);
    llh = my_em_censor_loglik(XX{h}, sampleparam{h}.mu, ...
                              sampleparam{h}.sig, xpattern{h});
    loglik = loglik + llh;
end

% 
ll_old = -Inf;
ll_hist = zeros(maxIt,1);

% ############# EM #################
for it = 1:maxIt
    % compute the log-likelihood
    
    % check convergence
    lldiff = loglik - ll_old;
    ll_hist(it) = loglik;
    if (lldiff >= 0 && lldiff < tol*abs(loglik))
        break;
    end
    ll_old = loglik;
    loglik = 0;
    
    % EM
    % compute sufficient statistics
    for h=1:H,
        [xhat_e, Q_e, alpha] = xhatQ(XX{h}, sampleparam{h}.mu, sampleparam{h}.sig, xpattern{h});
        % Numerical issue
        % when alpha is below the machine precision, 
        % alpha=0, xhat=inf and Q=nan
        alpha0 = (alpha==0);
        xhat = xhat_e;
        xhat(alpha0,:) = 0;
        Q = Q_e;
        Q(:,:,alpha0) = 0;
        % UPDATE MEAN
        sampleparam{h}.mu = mean(xhat);
        % UPDATE COV
        xhatc = bsxfun(@minus,xhat,sampleparam{h}.mu);
        sig0 = (xhatc'*xhatc + squeeze(sum(Q,3)))/sampleparam{h}.n;
        sampleparam{h}.sig = (sig0+sig0')/2 + regVal*eye(d);
        % UPDATE LIKELIHOOD
        llh = my_em_censor_loglik(XX{h}, sampleparam{h}.mu, ...
                              sampleparam{h}.sig, xpattern{h});
        loglik = loglik + llh;
    end
end

mu = zeros(1,d);
sigma = zeros(d);
delta = zeros(d);
M = zeros(d);
for h=1:H,
    mu = mu + sampleparam{h}.n*sampleparam{h}.mu/N;
    delta = delta + sampleparam{h}.n*sampleparam{h}.sig/N;
end
for h=1:H,
    M = M + sampleparam{h}.n*(sampleparam{h}.mu - mu)*(sampleparam{h}.mu - mu)'/N;
end
sigma=delta+M;
sampleparam{H+1}.mu=mu;
sampleparam{H+1}.delta=delta;
sampleparam{H+1}.sigmag=tospd(sigma)+regVal*eye(d);
sampleparam{H+1}.data = [y,x];

% output
param.iters = it;
param.log_lh = loglik;
param.ll_hist = ll_hist(1:it);
param.regVal = regVal;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xhat, Q, alpha] = xhatQ(x, mu, sig, pattern)

[N,d] = size(x);
x0 = bsxfun(@minus, x, mu);

xhat = x;
Q = zeros(d,d,N);
alpha = ones(N,1);

pttn = pattern.uniq_censored;
cnt = pattern.count;
xpttn = pattern.xpttn;
xR = pattern.range;

for ii=1:length(cnt)
    on = pttn(ii,:);
    mn = ~on;
%     dm = sum(mn);
    idx = find(xpttn(:,ii));
    
    % check censored element
    if all(on)
        continue
    end
    
    [R,f] = chol(sig(on,on));
    
    % mean and covariance of conditional normal 
    mu_mo = (x0(idx,on) / R) * (sig(mn,on) / R)';
    mu_mo = bsxfun(@plus, mu_mo, mu(mn));
    
    sigRinv = sig(mn,on) / R;
    sig_mo = sig(mn,mn) - sigRinv*sigRinv';
    sig_mo = (sig_mo + sig_mo')/2;
    
    % mean and covariance of truncated normal 
    xRidx = xR(idx,:);
    xRl = cell2mat(xRidx(:,1));
    xRu = cell2mat(xRidx(:,2));
    [tmu, tcov, talpha] = tmvn_m3(mu_mo, sig_mo, xRl, xRu);
    
    
    % replace censored elements with conditional mean
    xhat(idx,mn) = tmu;
    % covariance corrections for censored elements
    Q(mn,mn,idx) = tcov;
    % probability mass in the truncated region
    alpha(idx) = talpha;
end

end



