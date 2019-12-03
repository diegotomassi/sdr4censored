function MLEparam = PFCsample2mle(param,beta)
H=length(param);
MLEparam=param;
[p,d] =size(beta);
        aux = inv(param{H}.sigmag) + beta*inv(beta'*param{H}.delta*beta)*beta' - ...
              beta*inv(beta'*param{H}.sigmag*beta)*beta';
        MLEparam{H}.delta = inv(aux); % ####### MLE ##########
        aux = beta*inv(beta'*MLEparam{H}.delta*beta)*beta'*MLEparam{H}.delta;
        M = param{H}.sigmag - param{H}.delta;
        MLEparam{H}.sigmag = MLEparam{H}.delta + aux'*M*aux;
        mu = param{H}.mu(:);
        delta = MLEparam{H}.delta;
        N = 0;
        z = zeros(d,1);
        for h=1:(H-1),
            muy = param{h}.mu(:);
            ny = param{h}.n;
            zy = beta'*(muy-mu);
            z = z + ny*zy;
            N = N + ny;
        end
        z = z/N;
        for h=1:(H-1),
            MLEparam{h}.sig = MLEparam{H}.delta + ...
                              aux'*param{h}.sig*aux - aux'*param{H}.delta*aux;
            muy = param{h}.mu(:);
            zy = beta'*(muy-mu);
            au = inv(beta'*delta*beta);
            nuy = au*(zy - z);
            MLEparam{h}.mu = mu' + (delta*beta*nuy)';
        end