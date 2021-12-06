tic
%For generating holes based on Bar FK
rmin=.1;
rmax=.8;
a=-2.75;
N=800;

%Inverse cdf
transf = @(x) ((rmax^(1 + a) - rmin^(1 + a))*(rmin^(1 + a)/(rmax^(1 + a) - rmin^(1 + a)) + x)).^(1/(1 + a));

%Test
% test = transf(rand(1,100000));
% histogram(test);

holes = zeros(N,N);
[xmat,ymat] = meshgrid(1:N,1:N);
while sum(holes,'all')/N^2<=.0184
    x = randi([1 N]);
    y = randi([1 N]);
    r = transf(rand());
    newhole = sqrt((x-xmat).^2+(y-ymat).^2)<=(r^2/.2^2);
    if (sum(newhole,'all')>0)&&(sum(newhole.*holes,'all')==0)
        holes = holes + newhole;
    end
end

holes = (~holes)*1;

imagesc(holes);
colorbar
axis off;
pbaspect([1 1 1]);

toc

holesFile = reshape(holes,size(holes,1)*size(holes,2),1);

% Save data
text = ['holes' num2str(size(holes,1)) '_1.dat'];
save(text, 'holesFile', '-ascii')
