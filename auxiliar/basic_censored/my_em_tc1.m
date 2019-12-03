function param = my_em_tc1(x,y, xtb, xcb, init)
% gaussian mixture model em algorithm for censored data
% 
% x         N*d     censored observations
% xtb       2*d     truncation bounds [lower ; upper]
% xcb       2*d     censoring bounds [lower ; upper]
% init      struct  initial parameters
% 

if nargin<3 || ~exist('xtb', 'var')
    xtb = [-Inf; Inf];
end

if nargin<4 || ~exist('xcb', 'var')
    xcb = [-Inf; Inf];
end


% const
[N,d] = size(x);

regVal = eps(max(eig(cov(x))));        % a small number added to diagonal of cov mtx
tol = 1e-10;            % termination tolerance
maxIt = 100*d;          % max number of iterations


% cencoring pattern
y = grp2idx(y)-1;
x1 = x(logical(y),:);
x0 = x(~logical(y),:);
pattern1 = my_em_censor_pattern(x1, xcb);
pattern0 = my_em_censor_pattern(x0, xcb);


% init
if nargin < 4,  init = struct([]);  end
mu0 = mean(x0);
sig0 = get_cov(x0); 
n0 = size(x0,1);
mu1 = mean(x1);
sig1 = get_cov(x1); 
n1 = size(x1,1);
mu = (n1*mu1+n0*mu0)/N;
Delta = (n1*sig1+n2*sig2)/n;
Sigma = Delta + (n1*(mu1-mu)*(mu1-mu)'+n0*(mu0-mu)*(mu0-mu)')/N;

% 
ll_old = -Inf;
ll_hist = zeros(maxIt,1);

% EM
for it = 1:maxIt
    % compute the log-likelihood
    [ll0] = my_em_censor_post(x0, mu0, sig0, pattern0);
    [ll1] = my_em_censor_post(x1, mu1, sig1, pattern1);
    ll = ll0+ll1;
    
    % check convergence
    lldiff = ll - ll_old;
    ll_hist(it) = ll;
    if (lldiff >= 0 && lldiff < tol*abs(ll))
        break;
    end
    ll_old = ll;
    
    
    
    % EM
    % compute sufficient statistics
    [xhat_e, Q_e, alpha] = xhatQ(x0, mu0, sig0, pattern0);
        
        % Numerical issue
        % when alpha is below the machine precision, 
        % alpha=0, xhat=inf and Q=nan
        alpha0 = (alpha==0);
        xhat = xhat_e;
        xhat(alpha0,:) = 0;
        Q = Q_e;
        Q(:,:,alpha0) = 0;
        mu0 = mean(xhat);
        xhatc = bsxfun(@minus,xhat,mu0);
        sig0 = xhatc'*xhatc + squeeze(sum(Q,3));
        sig0 = (sig0+sig0')/2 + regVal*eye(d);

    [xhat_e, Q_e, alpha] = xhatQ(x1, mu1, sig1, pattern1);
        
        % Numerical issue
        % when alpha is below the machine precision, 
        % alpha=0, xhat=inf and Q=nan
        alpha0 = (alpha==0);
        xhat = xhat_e;
        xhat(alpha0,:) = 0;
        Q = Q_e;
        Q(:,:,alpha0) = 0;
        mu1 = mean(xhat);
        xhatc = bsxfun(@minus,xhat,mu1);
        sig1 = xhatc'*xhatc + squeeze(sum(Q,3));
        sig1 = (sig1+sig1')/2 + regVal*eye(d);
        
        mu = (n1*mu1+n0*mu0)/N;
        Delta = (n1*sig1+n2*sig2)/n;
        Sigma = Delta + (n1*(mu1-mu)*(mu1-mu)'+n0*(mu0-mu)*(mu0-mu)')/N;
end


% # of parameters
nParam = (d-1) + (d + d*(d+1)/2);

% output
param.mu{1} = mu0;
param.mu{2} = mu1;
param.sig{1} = sig0;
param.sig{2} = sig1;
param.n{1} = n0;
param.n{2} = n1;
param.iters = it;
param.log_lh = ll;
param.AIC = -2*ll + 2*nParam;
param.BIC = -2*ll + nParam*log(N);
param.ll_hist = ll_hist(1:it);
param.regVal = regVal;

end



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


function param0 = em_init(x, K, init, pattern)

[~,d] = size(x);

cmpl = pattern.complete;

if isempty(init)
    xc = x(cmpl,:);
    [idx, cent] = kmeans(xc, K);
    
    pp = zeros(1,K);
    mu = zeros(K,d);
    sig= zeros(d,d,K);
    for k=1:K
        pp(k) = sum(idx==k);
        mu(k,:) = cent(k,:);
        if pp(k)
            sig(:,:,k) = cov(xc(idx==k,:));
        else
            sig(:,:,k) = eye(d)*1e-10;
        end
    end
    
    param0.pp = pp;
    param0.mu = mu;
    param0.C = sig;
else
    param0 = init;
end


end

