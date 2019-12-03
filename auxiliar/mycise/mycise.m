function [fv,beta,st] = mycise(y,x,di,pw,parameters)
% This is the main function that achives coordinate-independent sparse estimation using
% either SIR or PFC for calculating M and N matrices. See references for details.
%    USAGE:
%  - outputs:
%    fv: the value of the objective function at the estimator
%    beta:  the estimator of the central subspace based on BIC criterion.
%    st: a vector with either 0 or 1 element to indicate which variable is
%    to be selected by CISE.

%  - inputs:
%    y: response vector.
%    x: predictors matrix.
%    di: the dimension of the central subspace
%    pw: the range of the penalty parameter of \lambda is from 0 to pw. 
%    method: model to calculate M and N matrices. So far, only 'PFC' or 'SIR' is accepted.
%
%
global ppp; % the number of predictor
global ddd; % the dimension of the central subspace
global nnn;
global MMM;
global NNN;

nnn = size(x,1);
ppp = size(x,2);
ddd= di;
nslice=6;

[MMM,NNN] = MN4pfc(parameters);
% [MMM,NNN] = MN4aida(parameters);

nin = pw*100+1;

for i=1:nin
    f = mybiccis(i/100-0.01);
    stop(i) = f;
end

[f,ind] = min(stop);

lap = ind/100-0.01;

for k = 1:19
    f= mybiccis(lap-0.01+k/1000);
    dstop(k) = f;
end

[f,dind] = min(dstop);

dlap = lap-0.01+dind/1000;

[fv, beta, st] = mybiccis(dlap);
