function x = mywt(b)

p0= length(b);

for i=1:p0
    x(i)=norm(b(i,:))^(-0.5);
end

