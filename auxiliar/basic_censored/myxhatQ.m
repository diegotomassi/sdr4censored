function [xhat, Q, alpha] = myxhatQ(x, mu, sig, pattern)

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
    
%     sig_mo
%     isposdef(sig_mo)
    
    
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
