
clear all; close all; clc;
warning off
% -------------------------------------------------------------------------
% parameter setting
c = 10;                             % regularization parameter for linear SVM in Liblinear package
mem_block = 3000;                   % maxmum number of testing features loaded each time  

% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');    % we use Liblinear package, you need to
                                % download and compile the matlab codes                   
tr_data_path = ['attribute_features/feat_apascal_train.mat'];    % directory of the original data
ts_data_path =['attribute_features/feat_apascal_test.mat'] ;
tr_label_path=['attribute_data/apascal_train.txt'];
ts_label_path=['attribute_data/apascal_test.txt'];
classname_path=['attribute_data/class_names.txt'];
attribute_path=['attribute_data/attribute_names.txt'];
fea_dir = 'features';           % directory for saving final image features

% -------------------------------------------------------------------------
% load the database
load(tr_data_path);
tr_data=feat;
clear feat
load(ts_data_path);
ts_data=feat;
clear feat
% -------------------------------------------------------------------------
%choose training and testing data, along with training the dictionary 

tr_num = size(tr_data, 2);                   
ts_num = size(ts_data,2);
nFea = tr_num + ts_num;
dFea = size(tr_data, 1);
tr_label = importfile(tr_label_path);
ts_label = importfile(ts_label_path);
clabel = importfile_row(classname_path);
nclass = length(clabel);

DictSize=30;
AEncoderMat=[];
ADictMat={};

%for jj=1:nclass,
 % rootpath=strcat('dictionary/',database.cname{jj});
 % mkdir(fullfile(rootpath));
%end

TrLabel=zeros(1,tr_num);
TsLabel=zeros(1,ts_num);
    for i=1:nclass
        idx=find(strcmp(tr_label(:,2),clabel{i}));
        TrLabel(idx)=i;
        clear idx
        idx=find(strcmp(ts_label(:,2),clabel{i}));
        TsLabel(idx)=i;
        clear idx
    end
fprintf('dictionary training...\n');
%train the dictionary 
[ ADictMat , AEncoderMat ] = dict_training(tr_data,TrLabel,DictSize,AEncoderMat,ADictMat);
fprintf('\ndone!\n');
%save and load
save('dictionary\ADictMat.mat', 'ADictMat');         %D
save('dictionary\AEnCoderMat.mat', 'AEncoderMat');   %P

load('dictionary\ADictMat.mat');
load('dictionary\AEnCoderMat.mat');
% -------------------------------------------------------------------------
% extract image features
fprintf('training number: %d\ntesting number: %d\ntotal:%d\n',tr_num,ts_num,nFea);
fprintf('getting features\n');

tr_fea = zeros(tr_num, dFea);
ts_fea = zeros(ts_num, dFea);

for iter1 = 1:tr_num,  
    if ~mod(iter1, 100),
       fprintf('.');
    end
    if ~mod(iter1, 1000),
        fprintf(' %d images processed\n', iter1);
    end
    label = TrLabel(iter1);
    %get feature
    Data = tr_data(:,iter1);
    PredictCoef = AEncoderMat*Data;
    i=label;
    tr_fea(jj, :)= (ADictMat{i}*PredictCoef((i-1)*DictSize+1:i*DictSize,:))';
    %tr_label(jj) = label;
end

for iter1 = 1:ts_num,  
    if ~mod(iter1, 100),
       fprintf('.');
    end
    if ~mod(iter1, 1000),
        fprintf(' %d images processed\n', iter1);
    end
    label = TsLabel(iter1);
    %get feature
    Data = ts_data(:,iter1);
    PredictCoef = AEncoderMat*Data;
    i=label;
    ts_fea(jj, :)= (ADictMat{i}*PredictCoef((i-1)*DictSize+1:i*DictSize,:))';
   % ts_label(jj) = label;
end
% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

attribute_name = importfile_name(attribute_path);
fprintf('\n Testing...\n');
    %get label
    len=size(tr_label,2)-6;
for ii=1:len,
    fprintf('attribute %d : "%s"\n',ii,attribute_name{ii});
    train_label=tr_label(:,ii+6);
    train_label=transpose(cell2mat(train_label'));
    options = ['-c ' num2str(c)];
    model = train(double(train_label), sparse(tr_fea), options);
    
    test_label=ts_label(:,ii+6);
    test_label=transpose(cell2mat(test_label'));
    % load the testing features
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        [C,accuracy,decision_values] = predict(test_label, sparse(ts_fea), model);
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        acc =zeros(3,num_block+1);
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            %curr_idx = ts_idx(block_idx); 
            
            % load the current block of features
            curr_ts_fea = ts_fea(block_idx,:);
            curr_ts_label = test_label(block_idx);
            % test the current block features
            %ts_label = [ts_label; curr_ts_label];
            [curr_C,curr_acc,decision_values] = predict(curr_ts_label, sparse(curr_ts_fea), model);
            C = [C; curr_C];
            acc(:,jj) = curr_acc;
        end
        jj=jj+1;
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = num_block*mem_block + (1:rem_fea);

        curr_ts_fea = ts_fea(curr_idx,:);
        curr_ts_label = test_label(curr_idx);
        
        [curr_C,curr_acc,decision_values] = predict(curr_ts_label, sparse(curr_ts_fea), model); 
        C = [C; curr_C];   
        acc(:,jj) = curr_acc;
        accuracy =mean(acc,2);
    end

clear train_label test_label idx 

    fprintf('\nClassification accuracy : \n');
    fprintf('%f\n',accuracy);
    fprintf('============\n')
end