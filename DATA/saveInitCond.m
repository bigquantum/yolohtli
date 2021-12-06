
clear all, close all, clc

%% Load data

% ALWAYS check path !!!!!!!!!!!!!!!!!!!!!!!!
pathh = './initCond/dataZ8/'; % <-----
rawDATA = importdata([pathh  'raw_data.dat']);

fid = fopen([pathh 'dataparamcsv.csv']);
parameters = textscan(fid,'%s%s','delimiter',',');

parameters{1,:}
tablehight = size(parameters{1,1},1);
table = zeros(tablehight,1);

for i = 3:tablehight(1,1)
    table(i) = str2num(cell2mat(parameters{:,2}(i)));
end

%%

nx = table(21);
ny = table(22);
U = reshape(rawDATA(:,1),nx,ny);
V = reshape(rawDATA(:,2),nx,ny);
W = reshape(rawDATA(:,3),nx,ny);

Uview = flip(reshape(rawDATA(:,1),nx,ny)',1); % transformation just for visualization
Vview = flip(reshape(rawDATA(:,2),nx,ny)',1); % transformation just for visualization
Wview = flip(reshape(rawDATA(:,3),nx,ny)',1); % transformation just for visualization

figure;
imagesc(Uview)
colorbar
title('A fucking spiral') 
xlabel('X')
ylabel('Y')
hold on

%%

% Parameters
zpixels = 1; % number of pixels in te extended volume

% Build 3D domain
DU3d1 = reshape(U,ny*nx*zpixels,1);
DV3d1 = reshape(V,ny*nx*zpixels,1);
DW3d1 = reshape(W,ny*nx*zpixels,1);

% Rewrite in my software format
Dout = cat(2,DU3d1,DV3d1,DW3d1);

% Save data
text = [pathh 'dataSpiral.dat'];
save(text, 'Dout', '-ascii')
