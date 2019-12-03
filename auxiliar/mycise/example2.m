%% for a discrete response

data =load('marcewithout.txt');
y = data(:,1); x= data(:,2:end);

% In this dataset the response is discrete. We look for a reduction to dimension dim=2
dim=2;
lambda_max = 10;
[fv,beta,st] = mycise4pfc(y,x,dim,lambda_max,'disc',[]);
proj = x*beta;

% plot the projection
figure; plotDR(proj,y,'disc','cise-pfc')

%% for a continuous response:
[n,p] = size(x);
sg = 0.01;
w = zeros(p,1); w([1:3]) = 1/sqrt(3); 
yy = x*w + sg*randn(n,1);

dim=1;
lambda_max = .001;
r = 2;
[fv,beta,st] = mycise4pfc(yy,x,dim,lambda_max,'cont',r);
proj = x*beta;

% plot the projection
figure; plotDR(proj,yy,'cont','cise-pfc')
