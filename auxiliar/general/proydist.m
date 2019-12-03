function out = proydist(a1,a2)
P1 = oproy(a1);
P2 = oproy(a2);
out = norm(P1-P2,'fro');

function p = oproy(a)
p = a*inv(a'*a)*a';