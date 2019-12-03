function out = pfc4censored(param,u)
h = length(param);
aux = invsqrtm(param{h}.delta);
S = aux*param{h}.sigmag*aux;
evecs = firsteigs(S,u);
out = orth(aux*evecs);
