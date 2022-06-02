% Aproximación discreta/Discrete Aproximation
clc; clear all; close all;

% Parámetros del modelo/Model Paremeters:

alpha = 0.36; % participación del capital en la producción
beta = 0.96; % factor de descuento
delta = 0.08; % tasa de depreciación
sigma = 1.5
rho = 0.95
sigma_e = 0.004

%% Inciso i): Método de Tauchen

%Método de Tauchen 

function [grid,P] = Tauchen(rho,N,sigma)

% Discretiza un proceso estocástico AR(1), dados sus parámetros

% Notación:

% rho = persistencia del proceso AR(1).
% N = número de puntos en la malla discreta.
% sigma = desviación estándar de las innovaciones.

% grid = malla del AR(1) discreto.
% P = matriz de transición.

        % Construyendo la malla de valores
        sigma_z = sqrt(sigma^2/(1-rho^2)); 
        step =(2*sigma_z*3)/(N-1); % distancia entre puntos
        grid = zeros(N,1); % inicializa la malla
        for i=1:N
            grid(i) = -3*sigma_z+ (i-1)*step; % obtiene cada punto de la malla
        end
        
        % Construyendo la matriz de transición
        
        P = zeros(N,N); % inicializa la matriz P
        if N>1 
           for i=1:N
            P(i,1)=normcdf((grid(1)+step/2-rho*grid(i))/sigma); % usando la distribución normal
            P(i,N)=1-normcdf((grid(N)-step/2-rho*grid(i))/sigma);
                for j=2:N-1
                P(i,j)=normcdf((grid(j)+step/2-rho*grid(i))/sigma) ...
                    -normcdf((grid(j)-step/2-rho*grid(i))/sigma);
                end
           end
        else
            P=1;
        end
        
end





% Definimos el número de puntos de la malla del choque
q = 5; 

% Llamamos al método de Tauchen
[zeta,P] = Tauchen(rho, q, sigma_e);


%% Inciso ii): Iteración de la función de valor

p = 200; % número de puntos de la malla

% Valor inicial para el capital y los precios
K = 31;
w = 2.2;
r = 0.04;
A = 0;

% Construimos una malla de puntos para el capital
amalla = linspace(0,70,p)';

% Definimos una matriz para la utilidad de todos los posibles estados
U  = zeros(p,q,p);

for i=1:p          % activos de hoy
    for j=1:q      % choque de hoy
        for l=1:p  % activos de mañana
            c = max(exp(zeta(j))*w + (1+r)*amalla(i) - amalla(l),1e-200); % valor del consumo
            U(i,j,l) = (c.^(1-sigma))./(1-sigma);  % valor de la utilidad
        end
    end
end

% Auxiliares para la iteración de la función de valor
V0 = zeros(p,q);
V1 = zeros(p,q);

termin = 0; % condición de término
iter = 1; % inicializa el número de iteraciones
maxit = 1000; % número máximo de iteraciones
crit = 1e-5; % criterio de tolerancia

% Auxiliar para las reglas de decisión óptimas
pol  = zeros(p,q);

% Algoritmo
while (termin==0 && iter<maxit)
  for i=1:p                               % activos de hoy
      for j=1:q                           % choque de hoy
          aux = zeros(p,1); % auxiliar para encontrar la utilidad máxima
          
          for l=1:p                       % activos de mañana
              aux(l) = U(i,j,l);
              
              for m=1:q                   % choque de mañana
                  aux(l) = aux(l)+ beta*P(j,m)*V0(l,m);
              end
              
          end
          
          [V1(i,j),pol(i,j)] = max(aux); % indexación lógica para obtener el máximo
      end
  end
  
  % Criterio de convergencia
  if norm(V0-V1) < crit
      termin = 1;
  end
  
  % Actualización de valores
  V0  = V1; 
  iter  = iter+1; 
end

% Calcumalos las reglas de decisión óptimas
K = zeros(p,q); % acumulación de activos
C = zeros(p,q); % consumo

for i=1:p                                  % activos de hoy
    for j=1:q                              % choque de hoy
        K(i,j) = amalla(pol(i,j));
        C(i,j) = exp(zeta(j))*w+(1+r)*amalla(i)-K(i,j);
    end
end 

figure(1)
subplot(3,1,1);
plot(amalla,K(:,1),'b',amalla,K(:,2),'r',amalla,K(:,3),'g',amalla,K(:,4),'y',amalla,K(:,5),'m')
title('Activos')
legend('\zeta_1 = -0.6','\zeta_2 = -0.3','\zeta_3 = 0','\zeta_4 = 0.3','\zeta_5 = 0.6');
xlabel('a')
ylabel('a''')

subplot(3,1,3);
plot(amalla,V1(:,1),'b',amalla,V1(:,2),'r',amalla,V1(:,3),'g',amalla,V1(:,4),'y',amalla,V1(:,5),'m')
legend('\zeta_1 = -0.6','\zeta_2 = -0.3','\zeta_3 = 0','\zeta_4 = 0.3','\zeta_5 = 0.6');
title('Función valor')
xlabel('a')
ylabel('V')

subplot(3,1,2);
plot(amalla,C(:,1),'b',amalla,C(:,2),'r',amalla,C(:,3),'g',amalla,C(:,4),'y',amalla,C(:,5),'m')
legend('\zeta_1 = -0.6','\zeta_2 = -0.3','\zeta_3 = 0','\zeta_4 = 0.3','\zeta_5 = 0.6');
title('Consumo')
xlabel('a')
ylabel('C')



% Regra de Decisión optima
% Trajectoria de consumo
% Función valor

%% Incisos iii): Simulación de la economía



















% HOW TO DO THIS?






% Número de períodos a simular
T = 10000; 

% Inicialozamos los vectores
zt  = zeros(T,1);   % indicadora de los choques    
at  = zeros(T+1,1); % valores de los choques
ai = zeros(T+1,1); % indicadora de los activos
ct  = zeros(T,1);
shock = zeros(T+1,1); % valores de los choques

% Le damos valores iniciales a las variables de estado
at(1) = 0;
ai(1) = 1;
zt(1) = 3;

% Vamos a simular los choques
pi_acum = cumsum(P',1)';
for t=1:T-1
    zt(t+1)=min(find(random('unif',0,1) <= pi_acum(zt(t),:)));
end

% Calculamos los valores de las variables del modelo
for t = 1:T
        ai(t+1) = pol(ai(t),zt(t));
        at(t+1) = amalla(ai(t+1));
        ct(t)   = C(ai(t),zt(t));
        shock(t)= zeta(zt(t));
end
    
% Obtenemos la distribución invariante

% Inicializamos la matriz que almacena la distribución
F = zeros(p,q);

% Auxiliar que indica el estado de la economía
state = zeros(p,q); 

for i=1:p % número de puntos en la malla de activos
    for j=1:q % número de puntos en la malla del choque
        for t=1001:10000 % no considera las primeras 1000 observaciones
            if ai(t)==i && zt(t)==j
                state(i,j) = state(i,j)+1; % aumenta la frecuencia de esa entrada
            end
        end
        F(i,j) = state(i,j)/9000; % divide los estados entre número de observaciones para obtener la distribución
    end
end

% Graficamos la distribución acumulativa para cada valor del choque
F = cumsum(F)./sum(F);

figure(2)
subplot(3,2,1);
plot(amalla,F(:,1))
title('\zeta_1 = -0.6')
xlabel('a');

subplot(3,2,2);
plot(amalla,F(:,2))
title('\zeta_2 = -0.3')
xlabel('a');

subplot(3,2,3);
plot(amalla,F(:,3))
title('\zeta_3 = 0')
xlabel('a');

subplot(3,2,4);
plot(amalla,F(:,4))
title('\zeta_4 = 0.3')
xlabel('a');

subplot(3,2,5);
plot(amalla,F(:,5))
title('\zeta_5 = 0.6')
xlabel('a');

%% Inciso 4) creatividad!


