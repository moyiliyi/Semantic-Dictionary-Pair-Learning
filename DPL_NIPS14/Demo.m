% =========================================================================
%   Shuhang Gu, Lei Zhang, Wangmeng Zuo and Xiangchu Feng,
%   "Projective Dictionary Pair Learning for Pattern Classification," In NIPS 2014.
%
%
%       Written by Shuhang Gu @ COMP HK-PolyU
%       Email: shuhanggu@gmail.com
%       Oct, 2014.
% =========================================================================

warning off
% Load training and testing data

%decaf
database = read_database_txt_dir('AwA-features-decaf\Animals_with_Attributes\Features\decaf');

clabel = unique(database.label);
nclass = length(clabel);
idx = [];
last=1;

for jj=1:nclass,
  rootpath=strcat('dictionary/',database.cname{jj});
  mkdir(fullfile(rootpath));
end

for jj = 1:nclass,%每一类一个字典
    idx_label = find(database.label == clabel(jj));
    num = length(idx_label);
    idx = [idx; idx_label];
    TrLabel = database.label(idx);
    a=database.path(idx);
    for ii = last:(num+last-1),  
        TrData(:,ii) = importfile(a{ii}, 1, 4096);
        %TrLabel(ii) = clabel(jj);
    end
    last=last+num;

TrLabel = TrLabel';

% Column normalization
TrData = normcol_equal(TrData);

%Parameter setting                
DictSize = 30;
tau    = 0.05;
lambda = 0.003;
gamma  = 0.0001;

% DPL trainig
[ DictMat , EncoderMat ] = TrainDPL(  TrData, TrLabel, DictSize, tau, lambda, gamma );

%save
 dpath=strcat('dictionary\',database.cname{jj},'\DictMat.mat');
 epath=strcat('dictionary\',database.cname{jj},'\EnCoderMat.mat');
 save(dpath, 'DictMat');
 save(epath,'EncoderMat');
    
    
%save('dictionary\DictMat.mat', 'DictMat');
%save('dictionary\EnCoderMat.mat', 'EncoderMat');

fprintf('accomplished %d\n',jj);
end