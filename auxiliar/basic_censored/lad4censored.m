function [Wn,fn,fp] = lad4censored(param,u)
%
% --------------------------- REFERENCES -----------------------------------
% Cook, R. D. and Forzani, L. (2009). Likelihood-based sufficient dimension reduction. 
% Journal of the American Statistical Association. 104 (485): 197-208.\\
% doi:10.1198/jasa.2009.0106.
% -------------------------REQUIRED PACKAGES--------------------------------
%  - SG_MIN PACKAGE: several functions to perform Stiefel-Grassmann optimization.
%
% ===============================================================================

%--- get handle to objective function and derivative ......................
Fhandle = F(@F4lad4censored,param);
dFhandle = dF(@dF4lad4censored,param);

%--- optimization .........................................................
p = cols(param.data)-1; Wn = eye(p);
fp = Fhandle(Wn);
if u == p,
    warning('LDR:nored','The subspace you are looking for has the same dimension as the original feature space')
    fn = fp;
else
    %--- get initial estimate .................................................
    if isempty(param.initvalue)||ischar(param.initvalue)
        guess = get_initial_estimate(param.data(1,:),param.data(:,2:end),u,data_parameters,parameters);
        Wo = guess(Fhandle);
    else
        Wo = param.initvalue;
    end
    [fn Wn] = sg_min(Fhandle,dFhandle,Wo,'prcg','euclidean',{1:u},'quiet',param.maxiter);
    Wn = orth(Wn);
end
