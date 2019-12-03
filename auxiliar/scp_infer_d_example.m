%% ---- we used dim=2 for the dimension of the reduction subspace. How can
% we infer this value form the data?
%
% using AIC
[dlad,dcmlad,dclad] = aicLAD4censored(X,Y,xcb);
disp(['Optimal dimension for cmLAD estimated using AIC is ',num2str(dcmlad)]);
disp(['Optimal dimension for cLAD estimated using AIC is ',num2str(dclad)]);
disp(['Optimal dimension for standard LAD estimated using AIC is ',num2str(dlad)]);


% the same using BIC
[dlad,dcmlad,dclad] = bicLAD4censored(X,Y,xcb);
disp(['Optimal dimension for cmLAD estimated using BIC is ',num2str(dcmlad)]);
disp(['Optimal dimension for cLAD estimated using BIC is ',num2str(dclad)]);
disp(['Optimal dimension for standard LAD estimated using BIC is ',num2str(dlad)]);
