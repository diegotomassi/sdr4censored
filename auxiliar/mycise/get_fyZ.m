function Fy = get_fyZ(Y)
n = length(Y);
Y = grp2idx(Y);
h = max(Y);
Fy = zeros(n,h-1);
for i=1:(max(Y)-1),
    idx = find(Y==i);
    ni = length(idx);
    Fy(:,i) = -ni/n;
    Fy(idx,i) = 1-ni/n;
end

    
    
        