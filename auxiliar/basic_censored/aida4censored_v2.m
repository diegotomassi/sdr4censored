function out = aida4censored_v2(sampleparam,u)
H = length(sampleparam) -1;
delta = sampleparam{H+1}.delta;
sigma = sampleparam{H+1}.sigmag;
aux = invsqrtm(delta);
nj = mycell2vec(sampleparam,'n',H);
N = sum(nj);
S = logm(aux*sigma*aux);
for h=1:H,
    S = S - nj(h)*logm(aux*sampleparam{h}.sig*aux)/N;
end
evecs = firsteigs(S,u);
out = orth(aux*evecs);
