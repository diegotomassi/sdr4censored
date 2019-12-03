function xcb = getLOD(X)
%% function xcb = getLOD(X)
%
% This function checks for limits of detection in each predictor in X and
% returns them in XCB. XCB is a 2*p matrix; the first row contains the
% Lower Limits of Detection and the second row the Upper Limits of
% Detection.
%
% =========================================================================
xmin = min(X);
xmax = max(X);
aux = [xmin;xmax];
for j=1:cols(X),
    if (length(find(X(:,j)==xmin(j)))==1),
        xmin(j) = -inf;
    end
 
    if (length(find(X(:,j)==xmax(j)))==1),
        xmax(j) = inf;
    end
end
xcb = [xmin; xmax];
 

