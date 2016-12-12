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
I_Mat    = eye(Dim,Dim);%������λ����
%�������Pk*�ĺ������������һ��
for i=1:ClassNum
    TempData      = Data(:,Label==i);%��Label==i�ж�Ӧ�������е�ֵ��ɵ�һ������
    DataMat{i}    = TempData;
    randn('seed',i);              %��ÿ����ͬi�������ֵ����ͬ          
    DictMat{i}    = normcol_equal(randn(Dim, DictSize));%�����ʼ��һ��normalized dictmat
    randn('seed',2*i);
    P_Mat{i}      = normcol_equal(randn(Dim, DictSize))';%p��dict��С��ת��
 
    TempDataC     = Data(:,Label~=i);%~=������
    DataInvMat{i} = inv(tau*TempData*TempData'+lambda*TempDataC*TempDataC'+gamma*I_Mat);%inv ���棻Pk*�ĺ���һ��
end

CoefMat = UpdateA(  DictMat, DataMat, P_Mat,  tau, DictSize  );

