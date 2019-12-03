function [ll] = my_em_censor_loglik(x, mu, sig, pattern)
% compute the log-likelihood
% 
N = size(x,1);
x0 = bsxfun(@minus, x, mu);
log_lh = zeros(N,1);
pttn = pattern.uniq_censored;
cnt = pattern.count;
xpttn = pattern.xpttn;
xR = pattern.range;
for ii=1:length(cnt)
    on = pttn(ii,:);
    mn = ~on;
    do = sum(on);
    dm = sum(mn);
    idx = find(xpttn(:,ii));
    % log likelihhod of observed x_o
    %     pause
    [R,~] = chol(sig(on,on));
    xRinv = x0(idx, on) / R;
    quadform = sum(xRinv.^2, 2);
    logSqrtDetSig = sum(log(diag(R)));
    
    % log prob mass of censored x_m given x_o
    if any(mn)
        mu_mo = (x0(idx,on) / R) * (sig(mn,on) / R)';
        mu_mo = bsxfun(@plus, mu_mo, mu(mn));
        
        sigRinv = sig(mn,on) / R;
        sig_mo = sig(mn,mn) - sigRinv*sigRinv';
        sig_mo = 0.5*(sig_mo + sig_mo'); % added
        
        %%%
        varless0 = diag(sig_mo) < eps;
        if any(varless0)
            sig_mo(diag(varless0)) = eps;
        end
        
        xRidx = xR(idx,:);
        xRl = cell2mat(xRidx(:,1));
        xRu = cell2mat(xRidx(:,2));
        log_Phi = log(mvncdf(xRl - mu_mo, xRu - mu_mo, zeros(1,dm), sig_mo));
    else
        log_Phi = 0;
    end
    
    % weighted log likelihood of x
    log_lh(idx) = -0.5*quadform + (-logSqrtDetSig + log_Phi) - do*log(2*pi)/2;
end
ll = sum(log_lh);
end


