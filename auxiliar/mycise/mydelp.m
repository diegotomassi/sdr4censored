function [x,y,ix] = delp(b,st)

global ppp;
global MMM;
global NNN;

t = 0;
s = 0;
y = st;


for i=1:ppp
    
   if st(i) == 0  
       continue
   else
       t = t +1;
       if norm(b(t,:)) <= 1e-6
           s = s+1;
           y(i)=0;
           ix(s)=t;
       else
           continue
       end
   end
end

if sum(y) < sum(st)
b(ix,:)=[];
MMM(:,ix)=[];
MMM(ix,:)=[];
NNN(:,ix)=[];
NNN(ix,:)=[];
x = b; 
else
    ix=0;
    x=b;
end
