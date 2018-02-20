function [FinalParameters,likelihood,StateVariables,StandardErrorsFromHessian,Tstatmodelo]=PreKalman(TaxasDeJuro,deltaT,tenors,parametrosInit)


parametros=parametrosInit;

% LimiteInferiorkappaP=[0.05,0,0;0,0.05,0;0,0,0.05];
% LimiteInferiorthetaP=[0,-1,-1];
% LimiteInferiorsigma=[.001,0,0;0,.001,0;0,0,.001];
% LimiteInferiorLambda=0.1;
% LimiteInferiorsigmaObs=0.001*ones(1,length(tenors));

LimiteInferiorkappaP=[0.1,0.1,0.1];
LimiteInferiorthetaP=[-0.1,-1,-1];
%LimiteInferiorthetaP=[0.0001,0.0001,0.0001];
LimiteInferiorsigma=[.001,.001,.001];
LimiteInferiorLambda=0.1;
LimiteInferiorsigmaObs=0.001*ones(1,length(tenors));

LimiteInferior=[LimiteInferiorkappaP(:);LimiteInferiorthetaP(:);LimiteInferiorsigma(:);LimiteInferiorLambda;LimiteInferiorsigmaObs(:)]';

% LimiteSuperiorkappaP=[1,0,0;0,1,0;0,0,1];
% LimiteSuperiorthetaP=[1,1,1];
% LimiteSuperiorsigma=[1,0,0;0,1,0;0,0,1];
% LimiteSuperiorLambda=2;
% LimiteSuperiorsigmaObs=ones(1,length(tenors));

LimiteSuperiorkappaP=[10,10,10];
LimiteSuperiorthetaP=[1,10,10];
LimiteSuperiorsigma=[1,1,1];
LimiteSuperiorLambda=10;
LimiteSuperiorsigmaObs=ones(1,length(tenors));

LimiteSuperior=[LimiteSuperiorkappaP(:);LimiteSuperiorthetaP(:);LimiteSuperiorsigma(:);LimiteSuperiorLambda;LimiteSuperiorsigmaObs(:)]';

options = struct('MaxFunEvals',3000);
[FinalParameters,fval,exitflag,output,constraints,grad,hessian] = fmincon(@(x) KalmanAFNS(x,deltaT,tenors,TaxasDeJuro),...
    parametros,[],[],[],[],LimiteInferior,LimiteSuperior,[],options);

StandardErrorsFromHessian=sqrt(diag(inv(hessian)))';
Tstatmodelo=FinalParameters'./StandardErrorsFromHessian;
likelihood=-fval;

% State Variables
StateVariablesaux=load('VariablesX');
StateVariables=StateVariablesaux.StateVariables;




