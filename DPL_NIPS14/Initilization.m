function [ DataMat, DictMat, P_Mat, DataInvMat, CoefMat ] = Initilization( Data , Label, DictSize, tau, lambda, gamma )
% In this intilization function, we do the following things:
% 1. Random initialization of dictioanry pair D and P for each class
% 2. Precompute the class-specific inverse matrix used in Eq. (10)
% 3. Compute matrix class-specific code matrix A by Eq. (8) 
%    with the random initilized D and P
%
% The randn seeds are setted to make sure the results in our paper are
% reproduceable. The randn seed setting can be removed, our algorithm is 
% not sensitive to the initilization of D and P. In most cases, different 
% initilization will lead to the same recognition accuracy on a wide randge
% of testing databases.


ClassNum = max(Label);
Dim      = size(Data,1);
I_Mat    = eye(Dim,Dim);%创建单位矩阵
%逐列算出Pk*的后面括号有逆的一项
for i=1:ClassNum
    TempData      = Data(:,Label==i);%第Label==i列对应的所有行的值组成的一个向量
    DataMat{i}    = TempData;
    randn('seed',i);              %让每次相同i的随机的值都相同          
    DictMat{i}    = normcol_equal(randn(Dim, DictSize));%随机初始化一个normalized dictmat
    randn('seed',2*i);
    P_Mat{i}      = normcol_equal(randn(Dim, DictSize))';%p和dict大小是转置
 
    TempDataC     = Data(:,Label~=i);%~=不等于
    DataInvMat{i} = inv(tau*TempData*TempData'+lambda*TempDataC*TempDataC'+gamma*I_Mat);%inv 求逆；Pk*的后面一项
end

CoefMat = UpdateA(  DictMat, DataMat, P_Mat,  tau, DictSize  );

