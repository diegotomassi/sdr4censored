function [f,beta,st] = biccis(lam)

global lambda;
global nnn;
global ddd;
global ppp;
global MMM;
global NNN;  

M0= MMM;
N0= NNN;

if lam < 0
    f = inf;
    return
end

lambda = lam;

[A,B] = mysqrt(NNN);
G = B*MMM*B;
[a2,b2]= eig(G);
[a1,b1] = sort(diag(b2));

if ddd == 2
x0 = B*[a2(:,b1(ppp)),a2(:,b1(ppp-1))];

elseif ddd==1
    x0 = B*a2(:,b1(ppp));
end

[beta,st] = myecis(x0);

MMM = M0;
NNN = N0;



%f =  -trace(beta'*M*beta)+ 2*d*(sum(st)-d)/n;
f = -trace(beta'*MMM*beta) + ddd*(sum(st)-ddd)*log(nnn)/nnn;
%f = p*log(trace(Sigma0)-trace(beta'*Sigma0*beta)) + 2*d*sum(st)*2/n;

%RIC
%pe = d*(sum(st)-d);
%f = -(n-pe)*trace(beta'*M*beta) + (n-pe)*log(n/(n-pe)) + pe*(log(n)-1) + 4/(n-pe-2);
