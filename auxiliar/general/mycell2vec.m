function out = mycell2vec(structure,field,K)
% K = length(structure);
out = zeros(K,1);
for k=1:K,
    out(k) = eval(strcat('structure{k}.',field));
end