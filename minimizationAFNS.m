function [kappaP,thetaP,sigma,lambda,sigmaObs,StateVariables]=minimizationAFNS(TaxasDeJuro,deltaT,tenors,iterations)
 

%nparametros=22+length(tenors);
nparametros=10+length(tenors);
parametros=zeros(1,nparametros);


%%%Maximizations of the loglikelihood
AllParameters=zeros(length(parametros),iterations);
AllLikelihoods=zeros(1,iterations);
AllStateVariables=zeros(size(TaxasDeJuro,1),3,iterations);
AllStandardErrorsFromHessian=zeros(length(parametros),iterations);
AllTStatmodelo=zeros(length(parametros),iterations);


for i=1:iterations
    
% kappaP=[(10-0.1)*rand+0.1,0,0;0,(10-0.1)*rand+0.1,0;0,0,(10-0.1)*rand+0.1];
% thetaP=[rand,(1-(-1))*rand-1,(1-(-1))*rand-1];
% sigma=[rand,0,0;0,rand,0;0,0,rand];
% lambda=(10-0.1)*rand+0.1;
% sigmaObs=rand*0.01*ones(1,length(tenors));

kappaP=[(10-0.1)*rand+0.1,(10-0.1)*rand+0.1,(10-0.1)*rand+0.1];
thetaP=[rand,(1-(-1))*rand-1,(1-(-1))*rand-1];
sigma=[rand,rand,rand];
lambda=(10-0.1)*rand+0.1;
sigmaObs=rand*0.01*ones(1,length(tenors));

parametros=[kappaP(:); thetaP(:); sigma(:); lambda; sigmaObs(:)];    
    
    tic
    i
    [AllParameters(:,i),AllLikelihoods(i),AllStateVariables(:,:,i),AllStandardErrorsFromHessian(:,i),AllTStatmodelo(:,i)]=...
    PreKalman(TaxasDeJuro,deltaT,tenors,parametros);
    toc
end

%Optimal Parameters
Index=find(AllLikelihoods==max(AllLikelihoods));

FinalParameters=AllParameters(:,Index);
MaxLikelihood=AllLikelihoods(Index);
StateVariables=AllStateVariables(:,:,Index);
StandardErrorsFromHessian=AllStandardErrorsFromHessian(:,Index);
TStatmodelo=AllTStatmodelo(:,Index);


% Model Fit
% kappaP=reshape(FinalParameters(1:9),3,3)
% thetaP=reshape(FinalParameters(10:12),1,3)
% sigma=reshape(FinalParameters(13:21),3,3)
% lambda=FinalParameters(22)
% sigmaObs=reshape(FinalParameters(23:end),1,length(tenors));

kappaP=diag(FinalParameters(1:3));
thetaP=FinalParameters(4:6)';
sigma=diag(FinalParameters(7:9));
lambda=FinalParameters(10);
sigmaObs=FinalParameters(11:end);


end