function [sumloglikelihood]=KalmanAFNS(parametros,deltaT,tenors,TaxasDeJuro)

kappaP=diag(parametros(1:3));
thetaP=parametros(4:6)';
sigma=diag(parametros(7:9));
lambda=parametros(10);
sigmaObs=parametros(11:end);

%for speed
loglikelihood=zeros(1,size(TaxasDeJuro,1));
Level=zeros(1,size(TaxasDeJuro,1));
Slope=zeros(1,size(TaxasDeJuro,1));
Curvature=zeros(1,size(TaxasDeJuro,1));


% number of tenors
ntenors=length(tenors);

% Matriz de factor loadings
FactorMatrix=zeros(ntenors,3);
FactorMatrix(:,1)=ones(ntenors,1);
FactorMatrix(:,2)=(ones(1,ntenors)-exp(-lambda*tenors))./(lambda*tenors);
FactorMatrix(:,3)=((ones(1,ntenors)-exp(-lambda*tenors))./(lambda*tenors))-exp(-lambda*tenors);

% Function with the affine term
Caux1 = @(lambda,T) (1/(2*(lambda^2)))-(1/(lambda^3))*((1-exp(-lambda*T))/T)+(1/(4*(lambda^3)))*((1-exp(-2*lambda*T))/T); 
Caux2 = @(lambda,T) (1/(2*(lambda^2)))+(1/(2*(lambda^2)))*exp(-lambda*T)-(1/(4*lambda))*T*exp(-2*lambda*T)-(3/(4*(lambda^2)))*exp(-2*lambda*T)...
    -(2/(lambda^3))*((1-exp(-lambda*T))/T)+(5/(8*(lambda^3)))*((1-exp(-2*lambda*T))/T);

C=@(lambda,sigma,T) sigma(1,1)*sigma(1,1)*((T^2)/6)+sigma(2,2)*sigma(2,2)*Caux1(lambda,T)+sigma(3,3)*sigma(3,3)*Caux2(lambda,T);

% Matrix Affine Term

AffineTermMatrix=zeros(ntenors,1);

for i=1:ntenors
 AffineTermMatrix(i)=C(lambda,sigma,tenors(i));   
end

% R matriz (varianceObservationErrors)
R=diag(sigmaObs);

% Transition Matrix (i.e, the A and B of the euqation X(t)=A+BX(t-1), where X is the state vector 
TransitionMatrixConstant=(eye(3)-expm(-kappaP*deltaT))*thetaP';
TransitionMatrixFactor=expm(-kappaP*deltaT);

% 1-Initialize the state vector
X=thetaP';
CSigma=diag(diag(sigma).*diag(sigma).*(diag(eye(3)-expm(-2*kappaP*deltaT))./diag(2*kappaP)));


% 2-Observable according to the model, variance and error
ModelObservable=-AffineTermMatrix+FactorMatrix*X;
ModelVariance=FactorMatrix*CSigma*FactorMatrix'+R;
ModelVariance=0.5*(ModelVariance+ModelVariance');

ErrorChi=TaxasDeJuro(1,:)'-ModelObservable;

% 3-Update valueof the state vectors given the error

KalmanGain=CSigma*FactorMatrix'*inv(ModelVariance);  %Kalman Gain Matrix
Xupdate=X+KalmanGain*ErrorChi; %update of the state vector
Xvarupdate=(eye(3)-KalmanGain*FactorMatrix)*CSigma; %update variance state vector

% 4-Forecast 2-moments state vector one step ahead
NextXmean=TransitionMatrixConstant+TransitionMatrixFactor*Xupdate; 
NextXvariance=TransitionMatrixFactor*Xvarupdate*TransitionMatrixFactor'+CSigma; 

% 5-first value for the log-likelihood
loglikelihood(1)=-0.5*ntenors*log(2*pi)-0.5*log(det(ModelVariance)+ErrorChi'*inv(ModelVariance)*ErrorChi);

% state variables
Level(1)=Xupdate(1);
Slope(1)=Xupdate(2);
Curvature(1)=Xupdate(3);

%%% repeat the steps for each observation
for i=2:size(TaxasDeJuro,1)
    
    % 2
    ModelObservable=-AffineTermMatrix+FactorMatrix*NextXmean;
    ModelVariance=FactorMatrix*NextXvariance*FactorMatrix'+R;  
    ModelVariance=0.5*(ModelVariance+ModelVariance');
    ErrorChi=TaxasDeJuro(i,:)'-ModelObservable;
   
    % 3
    KalmanGain=NextXvariance*FactorMatrix'*inv(ModelVariance);  
    Xupdate=NextXmean+KalmanGain*ErrorChi; 
    Xvarupdate=(eye(3)-KalmanGain*FactorMatrix)*NextXvariance; 
    
    % 4
    NextXmean=TransitionMatrixConstant+TransitionMatrixFactor*Xupdate;
    NextXvariance=TransitionMatrixFactor*Xvarupdate*TransitionMatrixFactor'+CSigma;
   
    % 5
    loglikelihood(i)=-0.5*ntenors*log(2*pi)-0.5*log(det(ModelVariance))-0.5*ErrorChi'*inv(ModelVariance)*ErrorChi;
    
    % state variables
    Level(i)=Xupdate(1);
    Slope(i)=Xupdate(2);
    Curvature(i)=Xupdate(3);
 
end

sumloglikelihood=-sum(loglikelihood);
StateVariables=[Level',Slope',Curvature'];
save('VariablesX','StateVariables');

end
