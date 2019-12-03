function [M,N] = MN4pfc_new(y,x,type,r)
if nargin < 3,
    error('No,t enough input arguments')
end

if strcmpi(type,'cont'),
    if nargin < 4,
        r = 3;
    end
    Fy = get_fy(y,r);
elseif strcmpi(type,'disc'),
    Fy = get_fyZ(y);
end
Xc = centering(x);
M = get_cov(Fy*inv(Fy'*Fy)*Fy'*Xc);
N = get_cov(Xc);


function xc = centering(x)
n = size(x,1);
xc = x - repmat(mean(x),n,1);