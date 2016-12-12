function [matout]=normcol_equal(matin)
% solve the proximal problem 
% matout = argmin||matout-matin||_F^2, s.t. matout(:,i)=1
matout = matin./repmat(sqrt(sum(matin.*matin,1)+eps),size(matin,1),1);
%sum(A,1) 每列加
%eps==eps（1）浮点数的相对精度（以防=0）
%repat（A，n，m）把A扩充成大矩阵，纵向复制n遍，横向复制m遍
%size（A，1）A的行数

