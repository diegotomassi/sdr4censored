function [M,N] = MN4pfc(param)
H=length(param)-1;
S = param{H+1}.sigmag;
mu = param{H+1}.mu(:);

Sfit = zeros(size(S));
nn = 0;
for h=1:H,
    Sfit = Sfit + param{h}.n*(param{h}.mu(:) - mu)*(param{h}.mu(:) - mu)';
    nn = nn + param{h}.n;
end
Sfit = Sfit/nn;

M = Sfit;
N = S;
