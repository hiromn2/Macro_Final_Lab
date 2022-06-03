% Aproximación discreta/Discrete Aproximation
% -----------------------------------------------------------------------------------------
clc; clear all; close all;

% Parametros del modelo/Model parameters:

alpha = 0.33; % participacion del capital en la produccion/particpation of capital in production 
beta = 0.96; % factor de descuento/discount factor 
delta = 0.08; % tasa de depreciacion del capital/capital depreciation rate
sigma = 1.5; % coeficiente de aversion al riesgo/risk aversion coefficient
rho = 0.95; % persistencia del choque estocastico/persistence of the stochastic shocks
sigma_e = 0.004; % desviacion estandar de las innovacion/standar deviation of innovation






%% Inciso i): Metodo de Tauchen/Tauchen method 

% Definimos el número de puntos de la malla del choque/ We define the number of points from the shock mesh
q = 5; 
%Lo anterior nos devuelve un vector con 5 valores que son en este caso/ The above returns a vector with 5 values which are in this case
%-0.04, -0.02, 0, 0.02 y 0.04 y nos va a dar una maiz p que será la matriz de /The vector returns a vector with 5 values which are: %-0.04, -0.02, 0, 0.02 and 0.04 and it will give us
%transición, la cual nos dice, estando en dichos valores, cuál es la/ a matrix p which will be the matrix of
%probabilidad de que pase a los otros estados. /which tells us, being at those values, what is the probability of passing to the other states. 




% Llamamos al método de Tauchen/We call the Tauchen method
[z,P] = Tauchen(rho, q, sigma_e); %P será nuestra matriz de transición/P will be our transition matrix

%% Inciso ii): Iteración de la función de valor /Value function iteration

% Valor inicial para el capital y los precios
K = 31; %Capital inicial
w = 2; 
r = 0.15; 
A = 0; %Valor inicial de productividad o el valor del choque inicialmente

% Creamos una malla para los activos
p = 200;
amalla = linspace(0,4,p)';  %Los activos van de 0 a 4

% Definimos una matriz para la utilidad de todos los posibles estados para los jóvenes
U1  = zeros(p,q,p); 

for i=1:p          % activos de hoy
    for j=1:q      % choque de hoy
        for l=1:p  % activos de mañana
            c1 = max(exp(z(j))*w + (1+r)*amalla(i) - amalla(l),1e-200); % valor del consumo en cada periodo
            U1(i,j,l) = (c1.^(1-sigma))./(1-sigma);  % valor de la utilidad
        end
    end
end

% Definimos un vector para la utilidad de los agentes retirados para
% diferentes activos 
U2  = zeros(p,q);
for i=1:p
    for j=1:q
        c2 = max((1+r)*amalla(i),1e-200);
        U2(i,j) = (c2.^(1-sigma))./(1-sigma);
    end
end

% Auxiliares para la iteraci�n de la funci�n de valor
V0 = zeros(p,q);
V1 = zeros(p,q);

% Auxiliar para la regla de decisión óptima
pol  = zeros(p,q);

termin = 0; % condición de término
iter = 1; % inicializa el número de iteraciones
maxit = 1000; % número máximo de iteraciones
crit = 1e-5; % criterio de tolerancia

% Auxiliar para las reglas de decisión óptimas
pol  = zeros(p,q); %Tendremos tanto el índice de los activos como el índice del choque estocástico

while (termin==0 && iter<maxit)
  for i=1:p                               % activos de hoy
      for j=1:q                           % choque de hoy
          aux = zeros(p,1); % auxiliar para encontrar la utilidad m�xima
          
          for l=1:p                       % activos de mañana
              aux(l) = U1(i,j,l);
              aux(l) = aux(l)+ beta*U2(l,j); 
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
C_young = zeros(p,q);
A_young = zeros(p,q);

for i=1:p
    for j=1:q
        A_young(i,j) = amalla(pol(i,j));
        C_young(i,j) = exp(z(j))*w+(1+r)*amalla(i)-amalla(pol(i,j));
    end
end

figure(1)
subplot(3,1,1);
plot(amalla,V1(:,1),'b',amalla,V1(:,2),'r',amalla,V1(:,3),'g',amalla,V1(:,4),'y',amalla,V1(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Función valor del agente joven')
xlabel('a')
ylabel('V')

subplot(3,1,2);
plot(amalla,C_young(:,1),'b',amalla,C_young(:,2),'r',amalla,C_young(:,3),'g',amalla,C_young(:,4),'y',amalla,C_young(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Consumo del agente joven')
xlabel('a')
ylabel('C')

subplot(3,1,3);
plot(amalla,A_young(:,1),'b',amalla,A_young(:,2),'r',amalla,A_young(:,3),'g',amalla,A_young(:,4),'y',amalla,A_young(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Activos')
xlabel('a')
ylabel('a''')

% Obtenemos y graficamos el consumo y la función valor del agente retirado
C_old = zeros(p,q);
V_old = zeros(p,q);

for i=1:p
    for j=1:q
        C_old = (1+r)*amalla(pol);
        V_old = (C_old.^(1-sigma))./(1-sigma);
    end
end

figure(2)
subplot(2,1,1);
plot(amalla,V_old(:,1),'b',amalla,V_old(:,2),'r',amalla,V_old(:,3),'g',amalla,V_old(:,4),'y',amalla,V_old(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Función valor del agente retirado')
xlabel('a')
ylabel('V')

subplot(2,1,2);
plot(amalla,C_old(:,1),'b',amalla,C_old(:,2),'r',amalla,C_old(:,3),'g',amalla,C_old(:,4),'y',amalla,C_old(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Consumo del agente retirado')
xlabel('a')
ylabel('C')


%% Incisos iii): Simulación de la economía
% Número de períodos a simular
T = 10000; 

% Inicialozamos los vectores
zt  = zeros(T,1);   % indicadora de los choques    
at  = zeros(T+1,1); % valores de los activos
ai = zeros(T+1,1); % indicadora de los activos
ct  = zeros(T,1);
shock = zeros(T+1,1); % valores de los choques
C = C_young

% Le damos valores iniciales a las variables de estado
at(1) = 0;
ai(1) = 1;
zt(1) = 3; %Indicadora del choque

% Vamos a simular los choques
pi_acum = cumsum(P',1)';
for t=1:T-1
    zt(t+1)=min(find(random('unif',0,1) <= pi_acum(zt(t),:)));
end
 
% Calculamos los valores de las variables del modelo
for t = 1:T
        ai(t+1) = pol(ai(t),zt(t)); %Buscamos el índice de mañana teniendo el índice de hoy
        at(t+1) = amalla(ai(t+1)); %Nos dice el valor del capital de mañana 
        ct(t)   = C(ai(t),zt(t)); %Nos dice cuál es el valor del consumo 
        shock(t)= z(zt(t)); %Nos dice cuál es el valor del schock
end

% Obtenemos la distribución invariante
% La distribución invariante estára dada a partir de todas las
% posibilidades queremos saber cuántas veces se cae en un estado en
% particular, se ve la función de distribución acumulada

% Inicializamos la matriz que almacena la distribución
F = zeros(p,q); %p*q es el número de estados que podemos tener 

% Auxiliar que indica el estado de la economía
state = zeros(p,q); 

for i=1:p % número de puntos en la malla de activos
    for j=1:q % número de puntos en la malla del choque
        for t=1001:10000 % no considera las primeras 1000 observaciones
            if ai(t)==i && zt(t)==j
                state(i,j) = state(i,j)+1; % aumenta la frecuencia de esa entrada
            end
        end
        F(i,j) = state(i,j)/9000; % divide los estados entre número de observaciones para obtener la distribuci�n
    end
end

% Graficamos la distribución acumulativa para cada valor del choque
% punto anterior
F = cumsum(F)./sum(F);

figure(3)
subplot(3,2,1);
plot(amalla,F(:,1))
title('z_1 = -0.04')
xlabel('a');

subplot(3,2,2);
plot(amalla,F(:,2))
title('z_2 = -0.02')
xlabel('a');

subplot(3,2,3);
plot(amalla,F(:,3))
title('z_3 = 0')
xlabel('a');

subplot(3,2,4);
plot(amalla,F(:,4))
title('z_4 = 0.02')
xlabel('a');

subplot(3,2,5);
plot(amalla,F(:,5))
title('z_5 = 0.04')
xlabel('a'); 













% Inciso 4) Changing the stochastic persistance parameter 

% Parámetros del modelo
alpha = 0.33; 
beta = 0.96; 
delta = 0.08; % tasa de depreciación del capital
sigma = 1.5; % coeficiente de aversión al riesgo
p = 200; % número de puntos de la malla
rho = 0.2; % persistencia del choque estocástico
sigma_e = 0.004; % desviación estándar de las innovaciones

%% Inciso i): Método de Tauchen

% Definimos el número de puntos de la malla del choque
q = 5; 
%Lo anterior nos devuelve un vector con 5 valores que son en este caso
%-0.04, -0.02, 0, 0.02 y 0.04 y nos va a dar una maiz p que será la matriz de
%transición, la cual nos dice, estando en dichos valores, cuál es la
%probabilidad de que pase a los otros estados. 

% Llamamos al método de Tauchen
[z,P] = Tauchen(rho, q, sigma_e); %P será nuestra matriz de transición

%% Inciso ii): Iteración de la función de valor

% Valor inicial para el capital y los precios
K = 31; %Capital inicial
w = 2; 
r = 0.15; 
A = 0; %Valor inicial de productividad o el valor del choque inicialmente

% Creamos una malla para los activos
p = 200;
amalla = linspace(0,4,p)';  %Los activos van de 0 a 4

% Definimos una matriz para la utilidad de todos los posibles estados para los jóvenes
U1  = zeros(p,q,p); 

for i=1:p          % activos de hoy
    for j=1:q      % choque de hoy
        for l=1:p  % activos de mañana
            c1 = max(exp(z(j))*w + (1+r)*amalla(i) - amalla(l),1e-200); % valor del consumo en cada periodo
            U1(i,j,l) = (c1.^(1-sigma))./(1-sigma);  % valor de la utilidad
        end
    end
end

% Definimos un vector para la utilidad de los agentes retirados para
% diferentes activos 
U2  = zeros(p,q);
for i=1:p
    for j=1:q
        c2 = max((1+r)*amalla(i),1e-200);
        U2(i,j) = (c2.^(1-sigma))./(1-sigma);
    end
end

% Auxiliares para la iteraci�n de la funci�n de valor
V0 = zeros(p,q);
V1 = zeros(p,q);

% Auxiliar para la regla de decisión óptima
pol  = zeros(p,q);

termin = 0; % condición de término
iter = 1; % inicializa el n�mero de iteraciones
maxit = 1000; % n�mero m�ximo de iteraciones
crit = 1e-5; % criterio de tolerancia

% Auxiliar para las reglas de decisión óptimas
pol  = zeros(p,q); %Tendremos tanto el índice de los activos como el índice del choque estocástico

while (termin==0 && iter<maxit)
  for i=1:p                               % activos de hoy
      for j=1:q                           % choque de hoy
          aux = zeros(p,1); % auxiliar para encontrar la utilidad m�xima
          
          for l=1:p                       % activos de ma�ana
              aux(l) = U1(i,j,l);
              aux(l) = aux(l)+ beta*U2(l,j); 
          end
          
          [V1(i,j),pol(i,j)] = max(aux); % indexación lógica para obtener el máximo
      end
  end
  
  % Criterio de convergencia
  if norm(V0-V1) < crit
      termin = 1;
  end
  
  % Actualizaci�n de valores
  V0  = V1; 
  iter  = iter+1; 
  
end

% Calcumalos las reglas de decisión óptimas, notemos que ya no son un
% vector, si no que se trata de una matriz de índices
C_young = zeros(p,q);
A_young = zeros(p,q);

for i=1:p
    for j=1:q
        A_young(i,j) = amalla(pol(i,j));
        C_young(i,j) = exp(z(j))*w+(1+r)*amalla(i)-amalla(pol(i,j));
    end
end

figure(1)
subplot(3,1,1);
plot(amalla,V1(:,1),'b',amalla,V1(:,2),'r',amalla,V1(:,3),'g',amalla,V1(:,4),'y',amalla,V1(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Función valor del agente joven')
xlabel('a')
ylabel('V')

subplot(3,1,2);
plot(amalla,C_young(:,1),'b',amalla,C_young(:,2),'r',amalla,C_young(:,3),'g',amalla,C_young(:,4),'y',amalla,C_young(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Consumo del agente joven')
xlabel('a')
ylabel('C')

subplot(3,1,3);
plot(amalla,A_young(:,1),'b',amalla,A_young(:,2),'r',amalla,A_young(:,3),'g',amalla,A_young(:,4),'y',amalla,A_young(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Activos')
xlabel('a')
ylabel('a''')

% Obtenemos y graficamos el consumo y la función valor del agente retirado
C_old = zeros(p,q);
V_old = zeros(p,q);

for i=1:p
    for j=1:q
        C_old = (1+r)*amalla(pol);
        V_old = (C_old.^(1-sigma))./(1-sigma);
    end
end

figure(2)
subplot(2,1,1);
plot(amalla,V_old(:,1),'b',amalla,V_old(:,2),'r',amalla,V_old(:,3),'g',amalla,V_old(:,4),'y',amalla,V_old(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Función valor del agente retirado')
xlabel('a')
ylabel('V')

subplot(2,1,2);
plot(amalla,C_old(:,1),'b',amalla,C_old(:,2),'r',amalla,C_old(:,3),'g',amalla,C_old(:,4),'y',amalla,C_old(:,5),'m');
legend('z_1 = -0.04','z_2 = -0.02','z_3 = 0','z_4 = 0.02','z_5 = 0.04');
title('Consumo del agente retirado')
xlabel('a')
ylabel('C')


%% Incisos iii): Simulación de la economía
% Número de períodos a simular
T = 10000; 

% Inicialozamos los vectores
zt  = zeros(T,1);   % indicadora de los choques    
at  = zeros(T+1,1); % valores de los activos
ai = zeros(T+1,1); % indicadora de los activos
ct  = zeros(T,1);
shock = zeros(T+1,1); % valores de los choques
C = C_young

% Le damos valores iniciales a las variables de estado
at(1) = 0;
ai(1) = 1;
zt(1) = 3; %Indicadora del choque

% Vamos a simular los choques
pi_acum = cumsum(P',1)';
for t=1:T-1
    zt(t+1)=min(find(random('unif',0,1) <= pi_acum(zt(t),:)));
end
 
% Calculamos los valores de las variables del modelo
for t = 1:T
        ai(t+1) = pol(ai(t),zt(t)); %Buscamos el índice de mañana teniendo el índice de hoy
        at(t+1) = amalla(ai(t+1)); %Nos dice el valor del capital de mañana 
        ct(t)   = C(ai(t),zt(t)); %Nos dice cuál es el valor del consumo 
        shock(t)= z(zt(t)); %Nos dice cuál es el valor del schock
end

% Obtenemos la distribución invariante/ We obtain the invariant distribution 
% La distribución invariante estára dada a partir de todas las/ The invariant distribution will be given by all 
% posibilidades queremos saber cuántas veces se cae en un estado en/ the posibilities we want to know how much times it is in a particular 
% particular, se ve la función de distribución acumulada/ state, we can see the cumulative distribution

% Inicializamos la matriz que almacena la distribución/ We let begin the distribution matrix
F = zeros(p,q); %p*q es el número de estados que podemos tener/ p*q will be the number of states we can have 

% Auxiliar que indica el estado de la economía/ Indicates de state of the economy
state = zeros(p,q); 

for i=1:p % número de puntos en la malla de activos/ number of points in the assets mesh
    for j=1:q % número de puntos en la malla del choque/ number of points in the mesh of the shock
        for t=1001:10000 % no considera las primeras 1000 observaciones/ doesn't consider the first 1000 observations
            if ai(t)==i && zt(t)==j
                state(i,j) = state(i,j)+1; % aumenta la frecuencia de esa entrada/ increases the entry frequency
            end
        end
        F(i,j) = state(i,j)/9000; % divide los estados entre número de observaciones para obtener la distribucion/divides the states by the number of observations
    end
end

% Graficamos la distribución acumulativa para cada valor del choque/ Ploting the cumulative distribution for each shock value
% punto anterior/ last point 
F = cumsum(F)./sum(F);

figure(3)
subplot(3,2,1);
plot(amalla,F(:,1))
title('z_1 = -0.04')
xlabel('a');

subplot(3,2,2);
plot(amalla,F(:,2))
title('z_2 = -0.02')
xlabel('a');

subplot(3,2,3);
plot(amalla,F(:,3))
title('z_3 = 0')
xlabel('a');

subplot(3,2,4);
plot(amalla,F(:,4))
title('z_4 = 0.02')
xlabel('a');

subplot(3,2,5);
plot(amalla,F(:,5))
title('z_5 = 0.04')
xlabel('a'); 





% FUNCION TAUCHEN/ TAUCHEN FUNCTION


function [grid,P] = Tauchen(rho,N,sigma)

% Discretiza un proceso estocástico AR(1), dados sus parámetros/ Discretizes an stochastic process AR(1) given its parameters

% Notación/Notation: 

% rho = persistencia del proceso AR(1)/persistance of the process AR(1).
% N = número de puntos en la malla discreta/number of points in the discrete mesh.
% sigma = desviación estándar de las innovaciones/ standard deviation of the innovations.

% grid = malla del AR(1) discreto/mesh of the discrete AR(1).
% P = matriz de transición/transition matrix.

        % Construyendo la malla de valores/ Building the mesh of values
        sigma_z = sqrt(sigma^2/(1-rho^2)); 
        step =(2*sigma_z*3)/(N-1); % distancia entre puntos/distance between points
        grid = zeros(N,1); % inicializa la malla/beginning og mesh
        for i=1:N
            grid(i) = -3*sigma_z+ (i-1)*step; % obtiene cada punto de la malla/obtains every point of the mesh
        end
        
        % Construyendo la matriz de transición/Building the transition mesh
        
        P = zeros(N,N); % inicializa la matriz P/ beginning of P matrix
        if N>1 
           for i=1:N
            P(i,1)=normcdf((grid(1)+step/2-rho*grid(i))/sigma); % usando la distribución normal/ using normal distribution
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

