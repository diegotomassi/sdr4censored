function [beta,sampleparam] = get_initial_estimate_censored(x,y,u,xcb,maxit)
if nargin < 5,
    maxit=10;
end
sampleparam = my_em_censored_p(x,y,xcb,maxit);
y=grp2idx(y);
nslices=max(y);
W = aida4censored_v2(sampleparam,u);
sampleparam{nslices+1}.initvalue = W;
beta = lad4censored_v2(sampleparam,u);
