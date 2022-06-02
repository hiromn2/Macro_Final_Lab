% Aproximación discreta/Discrete Aproximation
clc; clear all; close all;

% Parámetros del modelo/Model Paremeters:

alpha = 0.36; % participación del capital en la producción
beta = 0.96; % factor de descuento
delta = 0.08; % tasa de depreciación
sigma = 1.5
rho = 0.95
sigma_e = 0.004

%% Inciso i): Método de Tauchen/Tauchen Method







% Definimos el número de puntos de la malla del choque
q = 5; 

% Llamamos al método de Tauchen
[z,P] = Tauchen(rho, q, sigma_e);


%% Inciso ii): Iteración de la función de valor

p = 200; % número de puntos de la malla

% Valor inicial para el capital y los precios
K = 31;
w = 2.2;
r = 0.04;
A = 0;

% Construimos una malla de puntos para el capital
amalla = linspace(0,70,p)';


% NEW THINGS
V2 = zeros(p,1); % función valor del agente retirado
V1 = zeros(p,1); % función valor del agente joven

termin = 0; % indicador de término 
crit = 1e-5; % criterio de tolerancia
K0 = 0.16; % valor inicial

while (termin == 0)
    
  R = alpha*K0^(alpha-1) + (1-delta); % tasa de interés real
  W = (1-alpha)*K0^alpha; % salario
  
  for i=1:p
     V1(i,1) = (exp(z())*w-amalla(i)^(1-sigma))/(1-sigma) + beta*((R*amalla(i))^(1-sigma))/(1-sigma); % función valor del agente joven
  end
  
  [V,I] = max(V1); % encuentra el valor que maximiza V1
  K1 = amalla(I); % encuentra el capital que maximiza V1
  
  % Criterio de convergencia
  if abs(K0 - K1) < crit
      termin = 1;
  end
  
  % Actualización de valores
  K0  = (K0+K1)/2;
end



% NEW THINGS


% Definimos una matriz para la utilidad de todos los posibles estados
U  = zeros(p,q,p);

for i=1:p          % activos de hoy
    for j=1:q      % choque de hoy
        for l=1:p  % activos de mañana
            c = max(exp(z(j))*w + (1+r)*amalla(i) - amalla(l),1e-200); % valor del consumo
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
        C(i,j) = exp(z(j))*w+(1+r)*amalla(i)-K(i,j);
    end
end 

figure(1)
subplot(3,1,1);
plot(amalla,K(:,1),'b',amalla,K(:,2),'r',amalla,K(:,3),'g',amalla,K(:,4),'y',amalla,K(:,5),'m')
title('Activos')
legend('z_1 = -0.6','z_2 = -0.3','z_3 = 0','z_4 = 0.3','z_5 = 0.6');
xlabel('a')
ylabel('a''')

subplot(3,1,3);
plot(amalla,V1(:,1),'b',amalla,V1(:,2),'r',amalla,V1(:,3),'g',amalla,V1(:,4),'y',amalla,V1(:,5),'m')
legend('z_1 = -0.6','z_2 = -0.3','z_3 = 0','z_4 = 0.3','z_5 = 0.6');
title('Función valor')
xlabel('a')
ylabel('V')

subplot(3,1,2);
plot(amalla,C(:,1),'b',amalla,C(:,2),'r',amalla,C(:,3),'g',amalla,C(:,4),'y',amalla,C(:,5),'m')
legend('z_1 = -0.6','z_2 = -0.3','z_3 = 0','z_4 = 0.3','z_5 = 0.6');
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
z  = zeros(T,1);   % indicadora de los choques    
at  = zeros(T+1,1); % valores de los choques
ai = zeros(T+1,1); % indicadora de los activos
ct  = zeros(T,1);
shock = zeros(T+1,1); % valores de los choques

% Le damos valores iniciales a las variables de estado
at(1) = 0;
ai(1) = 1;
z(1) = 3;

% Vamos a simular los choques
pi_acum = cumsum(P',1)';
for t=1:T-1
    z(t+1)=min(find(random('unif',0,1) <= pi_acum(z(t),:)));
end

% Calculamos los valores de las variables del modelo
for t = 1:T
        ai(t+1) = pol(ai(t),z(t));
        at(t+1) = amalla(ai(t+1));
        ct(t)   = C(ai(t),z(t));
        shock(t)= zeta(z(t));
end
    
% Obtenemos la distribución invariante

% Inicializamos la matriz que almacena la distribución
F = zeros(p,q);

% Auxiliar que indica el estado de la economía
state = zeros(p,q); 

for i=1:p % número de puntos en la malla de activos
    for j=1:q % número de puntos en la malla del choque
        for t=1001:10000 % no considera las primeras 1000 observaciones
            if ai(t)==i && z(t)==j
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

% Inciso 4) Changin the stochastic persistance parameter 

% Parámetros del modelo

alpha = 0.36; 
beta = 0.96; 
delta = 0.08; % tasa de depreciación del capital
sigma = 1.5; % coeficiente de aversión al riesgo
p = 200; % número de puntos de la malla
rho = .2; % persistencia del choque estocástico
sigma_e = 0.0872; % desviación estándar de las innovaciones

%% Inciso i): Método de Tauchen

% Definimos el número de puntos de la malla del choque
q = 5; 
%Lo anterior nos devuelve un vector con 5 valores que son en este caso
%-0.6, -0.3, 0, 0.3 y 0.6 y nos va a dar una maiz p que será la matriz de
%transición, la cual nos dice, estando en dichos valores, cuál es la
%probabilidad de que pase a los otros estados. 

% Llamamos al método de Tauchen
[zeta,P] = Tauchen(rho, q, sigma_e); %P será nuestra matriz de transición

%% Inciso ii): Iteración de la función de valor

% Valor inicial para el capital y los precios
K = 31; %Capital inicial
w = 2.2;
r = 0.04;
A = 0; %Valor inicial de productividad o el valor del choque inicialmente

% Construimos una malla de puntos para el capital
amalla = linspace(0,70,p)'; %Los activos van de 0 a 70

% Definimos una matriz para la utilidad de todos los posibles estados
U  = zeros(p,q,p);

for i=1:p          % activos de hoy
    for j=1:q      % choque de hoy
        for l=1:p  % activos de mañana
            c = max(exp(zeta(j))*w + (1+r)*amalla(i) - amalla(l),1e-200); % valor del consumo en cada periodo
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
pol  = zeros(p,q); %Tendremos tanto el índice de los activos como el índice del choque estocástico

% Algoritmo
while (termin==0 && iter<maxit)
  for i=1:p                               % activos de hoy
      for j=1:q                           % choque de hoy
          aux = zeros(p,1); % auxiliar para encontrar la utilidad máxima
          
          for l=1:p                       % activos de mañana
              aux(l) = U(i,j,l); %Matriz U definida en 3 dimensiones
              
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

% Calcumalos las reglas de decisión óptimas, notemos que ya no son un
% vector, si no que se trata de una matriz de índices
K = zeros(p,q); % acumulación de activos
C = zeros(p,q); % consumo

for i=1:p                                  % activos de hoy
    for j=1:q                              % choque de hoy
        K(i,j) = amalla(pol(i,j));          % Evaluamos la malla en la regla de política óptima que son los índices
        C(i,j) = exp(zeta(j))*w+(1+r)*amalla(i)-K(i,j);
    end
end 

figure(1)
subplot(3,1,1);
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

subplot(3,1,3);
plot(amalla,K(:,1),'b',amalla,K(:,2),'r',amalla,K(:,3),'g',amalla,K(:,4),'y',amalla,K(:,5),'m')
title('Activos')
legend('\zeta_1 = -0.6','\zeta_2 = -0.3','\zeta_3 = 0','\zeta_4 = 0.3','\zeta_5 = 0.6');
xlabel('a')
ylabel('a''')

%% Incisos iii): Simulación de la economía
% Número de períodos a simular
T = 10000;  %10 mil periodos

% Inicialozamos los vectores
z  = zeros(T,1);   % indicadora de los choques    
at  = zeros(T+1,1); % valores de los choques
ai = zeros(T+1,1); % indicadora de los activos
ct  = zeros(T,1);
shock = zeros(T+1,1); % valores de los choques

% Le damos valores iniciales a las variables de estado (dichos valores son
% arbitrarios)
at(1) = 0;
ai(1) = 1;
z(1) = 3; %Indicadora del choque

% Vamos a simular los choques
pi_acum = cumsum(P',1)';
for t=1:T-1
    z(t+1)=min(find(random('unif',0,1) <= pi_acum(z(t),:)));
end %Nos da un vector con todos los choques posibles (en términos de índices)

%Para graficar los choques usamos:plot(zt(1000:15000)

% Calculamos los valores de las variables del modelo
for t = 1:T
        ai(t+1) = pol(ai(t),z(t)); %Buscamos el índice de mañana teniendo el índice de hoy
        at(t+1) = amalla(ai(t+1)); %Nos dice el valor del capital de mañana
        ct(t)   = C(ai(t),z(t)); %Nos dice cuál es el valor del consumo 
        shock(t)= zeta(z(t)); %Nos dice cuál es el valor del schock
end
    
% Obtenemos la distribución invariante
% La distribución invariante estára dada a partir de todas las
% posibilidades queremos saber cuántas veces se cae en un estado en
% particular, se ve la función de distribución acumulada

% Inicializamos la matriz que almacena la distribución
F = zeros(p,q); %p*q es el número de estados que podemos tener 

% Auxiliar que indica el estado de la economía
state = zeros(p,q); 

%Esta función itera sobre la malla de activos, sobre la malla del choque y
%va a ir rellenando con los óptimos encontrando los posibles equilibrios. 

for i=1:p % número de puntos en la malla de activos
    for j=1:q % número de puntos en la malla del choque
        for t=1001:10000 % no considera las primeras 1000 observaciones
            if ai(t)==i && z(t)==j %toma el índice óptimo y lo evalua con la iteración en la que se encuentra, es un estado
                state(i,j) = state(i,j)+1; % aumenta la frecuencia de esa entrada, le sumamos un 1
            end
        end
        F(i,j) = state(i,j)/9000; % divide los estados entre número de observaciones para obtener la distribución, nos da la probabilidad de que el estado óptimo ocurra de todos los estados posibles
    end
end

% Graficamos la distribución acumulativa para cada valor del choque del
% punto anterior

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






% DEFININDO TAUCHEN



function [grid,P] = Tauchen(rho,N,sigma)

% Discretiza un proceso estocástico AR(1), dados sus parámetros/ Discretizes an stochastic process AR(1) given its parameters

% Notación/Notation: 

% rho = persistencia del proceso AR(1)/persistance of the process AR(1).
% N = número de puntos en la malla discreta/number of points in the discrete mesh.
% sigma = desviación estándar de las innovaciones/ standard deviation of the innovations.

% grid = malla del AR(1) discreto/mesh of the discrete AR(1).
% P = matriz de transición/transition matrix.

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
