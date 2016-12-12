function [matout]=normcol_equal(matin)
% solve the proximal problem 
% matout = argmin||matout-matin||_F^2, s.t. matout(:,i)=1
matout = matin./repmat(sqrt(sum(matin.*matin,1)+eps),size(matin,1),1);
%sum(A,1) ÿ�м�
%eps==eps��1������������Ծ��ȣ��Է�=0��
%repat��A��n��m����A����ɴ����������n�飬������m��
%size��A��1��A������

