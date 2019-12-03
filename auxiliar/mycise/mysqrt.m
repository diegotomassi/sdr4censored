function [x,y] = mysqrt(N)

[a1,b1] = eig(N);

x = a1*diag(sqrt(diag(b1)))*a1';
x= (x+x')/2;

y = a1*diag(1./sqrt(diag(b1)))*a1';
y = (y+y')/2;