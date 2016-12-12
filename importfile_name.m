function attributenames = importfile_name(filename, startRow, endRow)
%IMPORTFILE1 ���ı��ļ��е���ֵ������Ϊ�����롣
%   ATTRIBUTENAMES = IMPORTFILE1(FILENAME) ��ȡ�ı��ļ� FILENAME ��Ĭ��ѡ����Χ�����ݡ�
%
%   ATTRIBUTENAMES = IMPORTFILE1(FILENAME, STARTROW, ENDROW) ��ȡ�ı��ļ�
%   FILENAME �� STARTROW �е� ENDROW ���е����ݡ�
%
% Example:
%   attributenames = importfile1('attribute_names.txt', 1, 64);
%
%    ������� TEXTSCAN��

% �� MATLAB �Զ������� 2015/11/05 15:33:19

%% ��ʼ��������
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% ÿ���ı��еĸ�ʽ�ַ���:
%   ��1: �ı� (%s)
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%s%[^\n\r]';

%% ���ı��ļ���
fileID = fopen(filename,'r');

%% ���ݸ�ʽ�ַ�����ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% �ر��ı��ļ���
fclose(fileID);

%% ���޷���������ݽ��еĺ���
% �ڵ��������δӦ���޷���������ݵĹ�����˲�����������롣Ҫ�����������޷���������ݵĴ��룬�����ļ���ѡ���޷������Ԫ����Ȼ���������ɽű���

%% �����������
attributenames = [dataArray{1:end-1}];
