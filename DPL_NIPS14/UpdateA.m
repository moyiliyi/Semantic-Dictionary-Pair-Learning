function [ Coef ] = UpdateA(  Dict, Data, P_Mat,  tau, DictSize )
% Update A by Eq. (8)

    ClassNum = size(Data,2);
    I_Mat    = eye(DictSize,DictSize);
    for i=1:ClassNum
        TempDict       = Dict{i};
        TempData       = Data{i};
        Coef{i}        = (TempDict'*TempDict+tau*I_Mat)\(TempDict'*TempData+tau*P_Mat{i}*TempData);%（8）Ak*表达式
    end
%x = A\B 用来求解线性方程 A*x = B.

