
clear all, close all, clc

%% Circular domain for PDE

n = 512;
m = 512;
Lx = 10;
Ly = 10;
x = linspace(-Lx/2,Lx/2,n);
y = linspace(-Ly/2,Ly/2,m);
[X,Y]=meshgrid(x,y);
dx = abs(x(1)-x(2));
% dy = abs(y(1)-y(2));
R = Lx/2;
circleLogic = (X).^2+(Y).^2 <= R^2;

figure(1)
imagesc(circleLogic)

data1 = reshape(circleLogic,n*m,1)*1;

% text = './cBoundary1024.dat';
% save(text, 'data1', '-ascii')

%% Circular domain for integral

Domain = reshape(circleLogic,n*m,1);
coeff = zeros(1,n*m);

for i = 2:(n*m-1)
   if Domain(i)==1.0
       coeff(i) = 2;
   end
   if  Domain(i)==1.0 && Domain(i-1)==0.0
       coeff(i) = 1.0;
   end
   if  Domain(i)==1.0 && Domain(i+1)==0.0
       coeff(i) = 1.0;
   end
end

coeff = reshape(coeff,n,m);
coeffTrapz = coeff.*coeff';

figure(2)
imagesc(coeffTrapz)

data2 = reshape(coeffTrapz,n*m,1);

% text = './cTrapz1024.dat';
% save(text, 'data2', '-ascii')



