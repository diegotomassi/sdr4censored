function [fv,beta,st] = mycise_new(y,x,di,pw,type,r,thr)
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
%    type: 'cont' for continuous responses or 'disc' for discrete responses
%    r (optional): degree of polynomial for the basis matrix Fy in the
%    inverse regression Xc|Fy
%    thr (optional): threshold value to set a whole row to 0
%
global ppp; % the number of predictor
global ddd; % the dimension of the central subspace
global nnn;
global MMM;
global NNN;
global TTT;

nnn = size(x,1);
ppp = size(x,2);
ddd= di;
if nargin < 7,
    TTT=1e-6;
else
    TTT=thr;
end

nslice=6;

[MMM,NNN] = MN4pfc_new(y,x,type,r);

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
