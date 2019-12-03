function [M,N] = MN4aida(param)
H = length(param)-1;
Sigma = param{H+1}.sigmag;
Delta = param{H+1}.delta;
Delta = (Delta+Delta')/2;
W = invsqrtm(Delta);
nn = 0;
for h=1:H,
    nn = nn+param{h}.n;
end
Saida = logm(W*Sigma*W);
for h=1:H,
    Saida = Saida - param{h}.n*logm(W*param{h}.sig*W)/nn;
end
M = Saida;
N = eye(cols(Delta));
