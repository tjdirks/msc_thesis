

clear
clc

%%flow: MainTermPremium->minimizationAFNS->PreKalman->KalmanAFNS

iterations=20;

[TaxasDeJuro,Datas]=xlsread('PortugalRates.xlsx','PTRatesMonthly');


%se full. Se não colocar como comentário
%Datas=Datas(2:end,1);

%Se small. Se não colocar como comentário
Datas=Datas(48:end,1);
TaxasDeJuro=TaxasDeJuro(47:end,:);

deltaT=1/12; %os dados são mensais 
tenors=[1/12,3/12,6/12,1,2,3,5,7,10,15,20,25,30];  

ntenors=length(tenors);
nobservations=size(TaxasDeJuro,1);

%rates used to calculate the risk premium
TermPremimFirstRate=1;
TermPremiumSecondRate=10;
NumberForecasts=TermPremiumSecondRate/TermPremimFirstRate; %quantas vezes uma das taxas cabe na outra
IndexFirstRate=find(tenors==TermPremimFirstRate);  %posição em coluna da primeira taxa
IndexSecondRate=find(tenors==TermPremiumSecondRate); %posição em coluna da segunda taxa

%chama outros códigos
[kappaP,thetaP,sigma,lambda,sigmaObs,StateVariables]=...
    minimizationAFNS(TaxasDeJuro,deltaT,tenors,iterations); %vai obter os parâmetros fazer nºminimizacoes=iterations e escolhendo a maxlikelihood

%serve para construir o risk premium 
RiskPremiumConstant=[kappaP(1,1)*thetaP(1)/sigma(1,1);kappaP(2,2)*thetaP(2)/sigma(2,2);kappaP(3,3)*thetaP(3)/sigma(3,3)];
RiskPremiumfactor=[-kappaP(1,1)/sigma(1,1),0,0;0,(lambda-kappaP(2,2))/sigma(2,2),0;0,0,(lambda-kappaP(3,3))/sigma(3,3)];

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%para fit do modelo%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

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

TaxasDeJuroModelo=zeros(size(TaxasDeJuro,1),size(TaxasDeJuro,2));
for i=1:nobservations
TaxasDeJuroModelo(i,:)=(-AffineTermMatrix+FactorMatrix*StateVariables(i,:)')';
end


figure(1)
vectorx = 1:1:size(StateVariables,1);
subplot(1,2,1)
%plot(StateVariables)
plotyy(vectorx,StateVariables(:,1),[vectorx',vectorx'],[StateVariables(:,2),StateVariables(:,3)]);
legend('Level','Slope','Curvature')
title('State Variables')
subplot(1,2,2)
plot(tenors,-10000*AffineTermMatrix)
legend('-AffineTerm(bp)')
set(gcf,'Color','w')
  
% Gráfico
Titulos=num2str(tenors);
figure(2)
for i=1:length(tenors)
subplot(3,5,i)
plot(100*TaxasDeJuro(:,i),'Color','k')
hold on
plot(100*TaxasDeJuroModelo(:,i),'Color','r')
legend('Real','Model')
end
set(gcf,'Color','w')

%%%%%%%%%%%%%%%%%%
%%%Term Premium%%%
%%%%%%%%%%%%%%%%%%
for j=1:nobservations    
    ForecastOneYearRate(j,1)=TaxasDeJuroModelo(j,IndexFirstRate);
    for i=1:NumberForecasts-1
    ForecastStateVariables=expm(-kappaP*i)*StateVariables(j,:)'+(eye(3)-expm(-kappaP*i))*thetaP';  
    ForecastOneYearRate(j,i+1)=FactorMatrix(IndexFirstRate,:)*ForecastStateVariables-AffineTermMatrix(IndexFirstRate);
    end
TermPremium(j)=TaxasDeJuroModelo(j,IndexSecondRate)-(1/NumberForecasts)*sum(ForecastOneYearRate(j,:));   
end

figure(3)
subplot(2,2,[1,2])
plot(1:1:nobservations,100*TermPremium)
legend('Term Premium')

%%%Forecast das taxas
subplot(2,2,[3,4])
plot(100*ForecastOneYearRate(:,1),'Color','g')
hold on
plot(100*ForecastOneYearRate(:,2),'Color','c')
hold on
plot(100*ForecastOneYearRate(:,10),'Color','r')
legend('Real','1y Forecast','10y forecast')
set(gcf,'Color','w')


