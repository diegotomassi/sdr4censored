% function [dlad,dcmlad,dclad] = bicLAD4censored(x,y,xcb,maxit)
%
% This function infers the dimension of the central subspace using BIC.
%
%
% INPUTS:
%
% X: predictor matrix
% Y: outcome (integer)
% XCB: limits of detection, as computes with getLOD()
% maxit: (optional) maximum number of iterations for the EM algorithm.
%
%
% OUTPUTS:
%
% dlad: dimension for standard LAD
% dcmlad: dimension for cmLAD
% dclad: dimension for cLAD.
%
%
%
%
% =========================================================================
function [dlad,dclad,dcmlad] = bicLAD4censored(x,y,xcb,maxit)

if nargin<4,
    maxit=30;
end

y = grp2idx(y);
H = max(y);
XX = cell(H,1);
xpattern = cell(H,1);

[n,p] = size(x);
dof = @(do) (p+(H-1)*do+do*(p-do)+(H-1)*do*(do+1)/2 + p*(p+1)/2);


for h=1:H,
    XX{h} = x(find(y==h),:);
    xpattern{h} = my_em_censor_pattern(XX{h},xcb);
end

bic_clad = zeros(p,1);
bic_cmlad = zeros(p,1);
bic_lad = zeros(p,1);

disp('STARTING BIC LOOP. THIS CAN TAKE QUITE A LONG TIME')
for u=1:p,
    disp(u);
    [LAD,cmLAD,cLAD] = my_em_censored_vLAD(x,y,u,xcb,maxit);
    llc=0;llcm=0;llst=0;
    for h=1:H,
        llc = llc + my_em_censor_loglik(XX{h},cLAD{h}.mu,cLAD{h}.sig,xpattern{h});
        llcm = llcm + my_em_censor_loglik(XX{h},cmLAD{h}.mu,cmLAD{h}.sig,xpattern{h});
        llst = llst + my_em_censor_loglik(XX{h},LAD{h}.mu,LAD{h}.sig,xpattern{h});
    end
    bic_clad(u) = -llc + log(n)*dof(u);
    bic_cmlad(u) = -llcm + log(n)*dof(u);
    bic_lad(u) = -llst + log(n)*dof(u);
end
dclad = argmin(bic_clad);
dcmlad = argmin(bic_cmlad);
dlad = argmin(bic_lad);