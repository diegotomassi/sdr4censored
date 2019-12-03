function out = mahaldist(delta,mu1,mu2)
dif = mu1(:)-mu2(:);
out = dif'*inv(delta)*dif;
