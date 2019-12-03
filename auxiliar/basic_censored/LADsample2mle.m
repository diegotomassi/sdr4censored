function MLEparam = LADsample2mle(sampleparam,beta)
MLEparam = sampleparam;
nslices = length(sampleparam)-1;
[p,d] = size(beta);
regVal = 5e-11*eye(p);

        auxDelta = beta*inv(beta'*sampleparam{nslices+1}.delta*beta)*beta';
        auxSigma = beta*inv(beta'*sampleparam{nslices+1}.sigmag*beta)*beta';
%         aux = inv(globalparam.sigmag) + beta*inv(beta'*globalparam.delta*beta)*beta' - ...
%               beta*inv(beta'*globalparam.sigmag*beta)*beta';
        invDelta = inv(sampleparam{nslices+1}.sigmag) + auxDelta - auxSigma;

        % UPDATE DELTA
        MLEparam{nslices+1}.delta = tospd(inv(invDelta)); % ####### MLE ##########
        
        % UPDATE SIGMA
        M = sampleparam{nslices+1}.sigmag - sampleparam{nslices+1}.delta;
        MLEparam{nslices+1}.sigmag = MLEparam{nslices+1}.delta + M;

        %         MLEparam{nslices+1}.sigmag = MLEparam{nslices+1}.delta + auxDelta*M*auxDelta';
%         aux = beta*inv(beta'*globalparam.delta*beta)*beta'*globalparam.delta;
%         globalparam.sigmag = globalparam.delta + aux'*M*aux;

        % UPDATE Delta_y
        delta = MLEparam{nslices+1}.delta;
%         auxDelta = beta*inv(beta'*delta*beta)*beta';
        mu = sampleparam{nslices+1}.mu(:);
        N = 0;
        z = zeros(d,1);
        for h=1:nslices,
            muy = sampleparam{h}.mu(:);
            ny = sampleparam{h}.n;
            zy = beta'*(muy-mu);
            z = z + ny*zy;
            N = N + ny;
        end
        z = z/N;
            
        for h=1:nslices,
            Th = tospd(sampleparam{h}.sig - delta);
            MLEparam{h}.sig = tospd(delta + delta*auxDelta*Th*auxDelta*delta);% + regVal;
            muy = sampleparam{h}.mu(:);
            zy = beta'*(muy-mu);
            au = inv(beta'*delta*beta);
            nuy = au*(zy - beta'*sampleparam{h}.sig*beta*au*z);
            MLEparam{h}.mu = mu' + (delta*beta*nuy)';
%             mu = mu + sampleparam{h}.mu * sampleparam{h}.n;
%             N = N + sampleparam{h}.n;
        end
        MLEparam{nslices+1}.output = beta;
        MLEparam{nslices+1}.mu = mu/N;

