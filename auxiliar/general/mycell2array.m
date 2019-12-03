function out = mycell2array(structure,field,K)
% K = length(structure); 
[p,q] = size(eval(strcat('structure{1}.',field)));
out = zeros(p,q,K);
for k=1:K,
    out(:,:,k) = eval(strcat('structure{k}.',field));
end