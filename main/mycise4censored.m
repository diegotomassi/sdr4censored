% This is the main function that achives coordinate-independent sparse estimation
% for sufficient dimension reduction.
%
%    USAGE:
%    [beta,st,fv] = mycise4censored(parameters,di,pw,thr,type)
%
%  - INPUTS:
%    parameters: cell-structure resulting from the analysis with LAD or PFC
%                  (see MY_EM_CENSORED_vLAD.m and MY_EM_CENSORED_vPFC.m).
%    di: the dimension of the central subspace
%    pw: the range of the penalty parameter of \lambda is from 0 to pw. 
%    thr: threshold value used to push a whole row of the estimate to 0.
%    method: model to calculate M and N matrices. So far, only 'PFC' or
%    'AIDA'.
%
%  - OUTPUTS:
%    - beta:  the estimator of the central subspace based on BIC criterion.
%    - st: a vector with either 0 or 1 element to indicate which variable is
%    to be selected by CISE.
%    fv: the value of the objective function at the estimator
% 
% 
% 
% 
% 
% 
% 
%
% ========================================================================
function [beta,st,fv] = mycise4censored(parameters,di,pw,thr,type)


global ppp; % the number of predictor
global ddd; % the dimension of the central subspace
global nnn;
global MMM;
global NNN;
global TTT;

N =0;
H = length(parameters)-1;
for j=1:H,
    N = N+parameters{j}.n;
end
nnn = N;
ppp = size(parameters{H+1}.sigmag,2);
ddd= di;
if nargin < 5,
    type = 'PFC';
end
if nargin < 4,
    TTT=1e-2;
else
    TTT=thr;
end

if strcmpi(type,'PFC'),
    [MMM,NNN] = MN4pfc(parameters);
else
    [MMM,NNN] = MN4aida(parameters);
end
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
beta = orth(beta);
