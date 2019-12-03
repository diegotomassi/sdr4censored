function [Wn,fn,fp] = lad4censored_v2(param,u)
%
% --------------------------- REFERENCES -----------------------------------
% Cook, R. D. and Forzani, L. (2009). Likelihood-based sufficient dimension reduction. 
% Journal of the American Statistical Association. 104 (485): 197-208.\\
% doi:10.1198/jasa.2009.0106.
% -------------------------REQUIRED PACKAGES--------------------------------
%  - SG_MIN PACKAGE: several functions to perform Stiefel-Grassmann optimization.
%
% ===============================================================================
H = length(param)-1;
Fparam.param=param(1:H);
Fparam.globalparam=param{H+1};
maxIter = 1000;

%--- get handle to objective function and derivative ......................
Fhandle = F(@F4lad4censored,Fparam);
dFhandle = dF(@dF4lad4censored,Fparam);

%--- optimization .........................................................
p = cols(Fparam.globalparam.delta); Wn = eye(p);
fp = Fhandle(Wn);
if u == p,
    warning('LDR:nored','The subspace you are looking for has the same dimension as the original feature space')
    fn = fp;
else
    %--- get initial estimate .................................................
    Wo = Fparam.globalparam.initvalue;
    [fn Wn] = sg_min(Fhandle,dFhandle,Wo,'prcg','euclidean',{1:u},'quiet',maxIter);
    Wn = orth(Wn);
end
fn = -fn;
fp = -fp;