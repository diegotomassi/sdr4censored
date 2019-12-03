function testList = getPart(n,kfold)
%
% function testList = getPart(n,kfold)
%
% This function returns a list of indices for the observations in each
% testing partition in a kfold cross validation procedure.
%
% Inputs:
% n : number of elements in the sample
% kfold: number of folds in the CV procedure.
%
% ====================================================================
aux = randperm(n);
blksize = floor(n/kfold);
for k=1:kfold,
    if k < kfold,
        idxTest = ((k-1)*blksize+1):(k*blksize);
        testList{k} = aux(idxTest);
    else
        idxTest = ((k-1)*blksize+1):n;
        testList{k} = aux(idxTest);
    end
end
