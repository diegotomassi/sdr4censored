%% LOAD THE DATA
setpaths;
datafile='newdata.txt';
data = load(datafile);
%%
% data = redDATAdec3;
Y = data(:,1); X = data(:,2:end);
[n,p] = size(X);

%% SOME EXPLORATORY ANALYSIS
%  
% % we sketch some scatter plots. PRESS a KEY after each plot to continue.
% for i=1:p,
%     for j=(i+1):p,
%         figure;
%         gscatter(X(:,i),X(:,j),Y);
%         pause(3);
%     end
% end
% pause
% close all;
%%
% we draw some histograms of the predictors, to get some idea of the
% censoring and check for transformations
for i=1:p
    figure;
    hist(X(:,i));
end
 
%% Transform de data if needed
 
%% SEARCH FOR LIMITS OF DETECTION
xcb = getLOD(X);
 
%% PERFORM DIMENSION REDUCTION

% -- using LAD methods
dim = 2;
[MLEparam,cmMLEparam,ladMLEparam] = my_em_censored_vLAD(X,Y,dim,xcb,25);

% projection using cLAD
W = MLEparam{3}.output;
projections_cLAD = X*W;
figure;
gscatter(projections_cLAD(:,1),projections_cLAD(:,2),Y);
xlabel('cLAD-1'); ylabel('cLAD-2');
title('Projections using cLAD');

% projection using cmLAD
W = cmMLEparam{3}.output;
projections_cmLAD = X*W;
figure;
gscatter(projections_cmLAD(:,1),projections_cmLAD(:,2),Y);
xlabel('cmLAD-1'); ylabel('cmLAD-2');
title('Projections using cmLAD');

% projection using LAD
W = ladMLEparam{3}.output;
projections_LAD = X*W;
figure;
gscatter(projections_LAD(:,1),projections_LAD(:,2),Y);
xlabel('LAD-1'); ylabel('LAD-2');
title('Projections using LAD');




%% ============ PREDICTION ASSESSMENT =============
 
%% Accuracy assessment
% We will use 10-fold Cross Validation to assess prediction accuracy. We
% compare only cLAD/cPFC and LAD/PFC.
rng(100);
dim = 2;
maxIt = 25;
kfold = 10;
testidx = getPART(n,kfold); % this finds the observation indices in each partition
 
%-- array initialization
ypred_clad_qda = zeros(n,1);
post_clad_qda = zeros(n,1);
ypred_clad_lda = zeros(n,1);
post_clad_lda = zeros(n,1);

ypred_cmlad_qda = zeros(n,1);
post_cmlad_qda = zeros(n,1);
ypred_cmlad_lda = zeros(n,1);
post_cmlad_lda = zeros(n,1);

ypred_lad_qda = zeros(n,1);
post_lad_qda = zeros(n,1);
ypred_lad_lda = zeros(n,1);
post_lad_lda = zeros(n,1);


ypred_pfc_lda = zeros(n,1);
post_pfc_lda = zeros(n,1);
ypred_cpfc_lda = zeros(n,1);
post_cpfc_lda = zeros(n,1);
ypred_cmpfc_lda = zeros(n,1);
post_cmpfc_lda = zeros(n,1);

ypred_lda = zeros(n,1);
post_lda = zeros(n,1);
ypred_qda = zeros(n,1);
post_qda = zeros(n,1);

post_pfc_lr = zeros(n,1);
post_cpfc_lr = zeros(n,1);
post_cmpfc_lr = zeros(n,1);
post_lad_lr = zeros(n,1);
post_clad_lr = zeros(n,1);
post_cmlad_lr = zeros(n,1);
post_lr = zeros(n,1);

% #### MAIN LOOP FOR CROSS VALIDATION #############
for k=1:kfold
    
    disp(['==RUNNING FOLD ',num2str(k),' out of ',num2str(kfold)])
    
    Xtest = X(testidx{k},:); Ytest = Y(testidx{k},:);
    Xtrain = X; Xtrain(testidx{k},:) = []; Ytrain=Y; Ytrain(testidx{k})=[];
    
    % find the reduction with the LAD methods....
    [MLEparam,cmMLE,ladMLE,globalparam] = my_em_censored_vLAD(Xtrain,Ytrain,dim,xcb,25);
    
    % cLAD
    alpha = MLEparam{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'quadratic','empirical');
    ypred_clad_qda(testidx{k})=yhat;
    erra_clad_qda(k)=ea;
    post_clad_qda(testidx{k})=post(:,2);
    err_clad_qda(k)=mean(yhat~=Ytest);
    
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_clad_lda(testidx{k})=yhat;
    erra_clad_lda(k)=ea;
    post_clad_lda(testidx{k})=post(:,2);
    err_clad_lda(k)=mean(yhat~=Ytest);
    
    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_clad_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_clad_lr(k) = mean(yhat~=Ytest);
    
    
    % cmLAD
    alpha = cmMLE{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'quadratic','empirical');
    ypred_cmlad_qda(testidx{k})=yhat;
    erra_cmlad_qda(k)=ea;
    post_cmlad_qda(testidx{k})=post(:,2);
    err_cmlad_qda(k)=mean(yhat~=Ytest);
    
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_cmlad_lda(testidx{k})=yhat;
    erra_cmlad_lda(k)=ea;
    post_cmlad_lda(testidx{k})=post(:,2);
    err_cmlad_lda(k)=mean(yhat~=Ytest);
    

    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_cmlad_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_cmlad_lr(k) = mean(yhat~=Ytest);

    % standard LAD
    alpha = ladMLE{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'quadratic','empirical');
    ypred_lad_qda(testidx{k})=yhat;
    erra_lad_qda(k)=ea;
    post_lad_qda(testidx{k})=post(:,2);
    err_lad_qda(k)=mean(yhat~=Ytest);
    
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_lad_lda(testidx{k})=yhat;
    erra_lad_lda(k)=ea;
    post_lad_lda(testidx{k})=post(:,2);
    err_lad_lda(k)=mean(yhat~=Ytest);
    
    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_lad_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_lad_lr(k) = mean(yhat~=Ytest);
    

    % ####### PFC ############################
    [MLEparam,cmMLEparam,stMLEparam] = my_em_censored_vPFC(X,Y,dim,xcb,25);

    % cPFC
    alpha = MLEparam{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_cpfc_lda(testidx{k})=yhat;
    erra_cpfc_lda(k)=ea;
    post_cpfc_lda(testidx{k})=post(:,2);
    err_cpfc_lda(k)=mean(yhat~=Ytest);

    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_cpfc_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_cpfc_lr(k) = mean(yhat~=Ytest);

    % cmPFC
    alpha = cmMLEparam{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_cmpfc_lda(testidx{k})=yhat;
    erra_cmpfc_lda(k)=ea;
    post_cmpfc_lda(testidx{k})=post(:,2);
    err_cmpfc_lda(k)=mean(yhat~=Ytest);

    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_cmpfc_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_cmpfc_lr(k) = mean(yhat~=Ytest);

    % standard PFC
    alpha = stMLEparam{3}.output;
    [yhat,ea,post]=classify(Xtest*alpha,Xtrain*alpha,Ytrain,'linear','empirical');
    ypred_pfc_lda(testidx{k})=yhat;
    erra_pfc_lda(k)=ea;
    post_pfc_lda(testidx{k})=post(:,2);
    err_pfc_lda(k)=mean(yhat~=Ytest);

    m = glmfit(Xtrain*alpha,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest*alpha,'logit');
    post_pfc_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_pfc_lr(k) = mean(yhat~=Ytest);

    % no reduction
    [yhat,ea,post]=classify(Xtest,Xtrain,Ytrain,'linear','empirical');
    ypred_lda(testidx{k})=yhat;
    erra_lda(k)=ea;
    post_lda(testidx{k})=post(:,2);
    err_lda(k)=mean(yhat~=Ytest);

    [yhat,ea,post]=classify(Xtest,Xtrain,Ytrain,'quadratic','empirical');
    ypred_qda(testidx{k})=yhat;
    erra_qda(k)=ea;
    post_qda(testidx{k})=post(:,2);
    err_qda(k)=mean(yhat~=Ytest);

    m = glmfit(Xtrain,Ytrain,'binomial','link','logit');
    poster = glmval(m,Xtest,'logit');
    post_lr(testidx{k}) = poster;
    yhat = poster > 0.5;
    err_lr(k) = mean(yhat~=Ytest);

end   

 %% Prediction error (pe)
 % with LAD methods
 pe_clad_lda = mean(err_clad_lda);
 pe_clad_qda = mean(err_clad_qda);
 pe_clad_lr = mean(err_clad_lr);
 
  pe_cmlad_lda = mean(err_cmlad_lda);
  pe_cmlad_lr = mean(err_cmlad_lr);
  pe_cmlad_qda = mean(err_cmlad_qda);
  
  pe_lad_lda = mean(err_lad_lda);
 pe_lad_qda = mean(err_lad_qda);
pe_lad_lr = mean(err_lad_lr);


 % with PFC methods
 pe_cpfc_lda = mean(err_cpfc_lda);
  pe_cpfc_lr = mean(err_cpfc_lr);
 
 pe_cmpfc_lda = mean(err_cmpfc_lda);
 pe_cmpfc_lr = mean(err_cmpfc_lr);
 
 pe_pfc_lda = mean(err_pfc_lda);
 pe_pfc_lr = mean(err_pfc_lr);
 
 % without reduction
pe_lda = mean(err_lda);
pe_qda = mean(err_qda);
pe_lr = mean(err_lr);
 
 %%
% ROC Analysis
 % LAD
 [~,~,~,AUC_clad_lda] = perfcurve(Y,post_clad_lda,1);
 [~,~,~,AUC_cmlad_lda] = perfcurve(Y,post_cmlad_lda,1);
 [~,~,~,AUC_lad_lda] = perfcurve(Y,post_lad_lda,1);
%
 [~,~,~,AUC_clad_qda] = perfcurve(Y,post_clad_qda,1);
 [~,~,~,AUC_cmlad_qda] = perfcurve(Y,post_cmlad_qda,1);
 [~,~,~,AUC_lad_qda] = perfcurve(Y,post_lad_qda,1);
%
[~,~,~,AUC_clad_lr] = perfcurve(Y,post_clad_lr,1);
 [~,~,~,AUC_cmlad_lr] = perfcurve(Y,post_cmlad_lr,1);
 [~,~,~,AUC_lad_lr] = perfcurve(Y,post_lad_lr,1);

 % PFC
 [~,~,~,AUC_cpfc_lda] = perfcurve(Y,post_cpfc_lda,1);
 [~,~,~,AUC_cmpfc_lda] = perfcurve(Y,post_cmpfc_lda,1);
 [~,~,~,AUC_pfc_lda] = perfcurve(Y,post_pfc_lda,1);
%
[~,~,~,AUC_cpfc_lr] = perfcurve(Y,post_cpfc_lr,1);
 [~,~,~,AUC_cmpfc_lr] = perfcurve(Y,post_cmpfc_lr,1);
 [~,~,~,AUC_pfc_lr] = perfcurve(Y,post_pfc_lr,1);

 % No reduction
 [~,~,~,AUC_lda] = perfcurve(Y,post_lda,1);
 [~,~,~,AUC_qda] = perfcurve(Y,post_qda,1);
 [~,~,~,AUC_lr] = perfcurve(Y,post_lr,1);

%% ARE ALL THE PREDICTORS RELEVANT?

% LAD;
[MLEparam,cmMLEparam,stMLEparam] = my_em_censored_vLAD(X,Y,dim,xcb,25);
[beta_clad,st] = mycise4censored(MLEparam,2,10,.001);
disp('--- Regularized estimate for CLAD')
disp(beta_clad)
disp(['Selected variables are: ',num2str(find(st)')]);

[beta_cmlad,st] = mycise4censored(cmMLEparam,2,10,.001)
disp('--- Regularized estimate for CMLAD')
disp(beta_cmlad)
disp(['Selected variables are: ',num2str(find(st)')]);

[beta_lad,st] = mycise4censored(stMLEparam,2,10,.001)
disp('--- Regularized estimate for standard LAD')
disp(beta_lad)
disp(['Selected variables are: ',num2str(find(st)')]);


% PFC;
[MLEparam,cmMLEparam,stMLEparam] = my_em_censored_vPFC(X,Y,1,xcb,25);
[beta_cpfc,st] = mycise4censored(MLEparam,2,10,.001)
disp('--- Regularized estimate for CPFC')
disp(beta_cpfc)
disp(['Selected variables are: ',num2str(find(st)')]);

[beta_cmpfc,st] = mycise4censored(cmMLEparam,2,10,.001)
disp('--- Regularized estimate for CMPFC')
disp(beta_cmpfc)
disp(['Selected variables are: ',num2str(find(st)')]);


[beta_pfc,st] = mycise4censored(stMLEparam,2,10,.001)
disp('--- Regularized estimate for standard PFC')
disp(beta_pfc)
disp(['Selected variables are: ',num2str(find(st)')]);
