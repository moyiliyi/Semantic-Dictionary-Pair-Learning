function apascaltrain = importfile(filename, startRow, endRow)
%IMPORTFILE 将文本文件中的数值数据作为矩阵导入。
%   APASCALTRAIN = IMPORTFILE(FILENAME) 读取文本文件 FILENAME 中默认选定范围的数据。
%
%   APASCALTRAIN = IMPORTFILE(FILENAME, STARTROW, ENDROW) 读取文本文件 FILENAME 的
%   STARTROW 行到 ENDROW 行中的数据。
%
% Example:
%   apascaltrain = importfile('apascal_train.txt', 1, 6340);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2015/11/04 22:27:54

%% 初始化变量。
delimiter = ' ';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% 每个文本行的格式字符串:
%   列1: 文本 (%s)
%	列2: 文本 (%s)
%   列3: 双精度值 (%f)
%	列4: 双精度值 (%f)
%   列5: 双精度值 (%f)
%	列6: 双精度值 (%f)
%   列7: 双精度值 (%f)
%	列8: 双精度值 (%f)
%   列9: 双精度值 (%f)
%	列10: 双精度值 (%f)
%   列11: 双精度值 (%f)
%	列12: 双精度值 (%f)
%   列13: 双精度值 (%f)
%	列14: 双精度值 (%f)
%   列15: 双精度值 (%f)
%	列16: 双精度值 (%f)
%   列17: 双精度值 (%f)
%	列18: 双精度值 (%f)
%   列19: 双精度值 (%f)
%	列20: 双精度值 (%f)
%   列21: 双精度值 (%f)
%	列22: 双精度值 (%f)
%   列23: 双精度值 (%f)
%	列24: 双精度值 (%f)
%   列25: 双精度值 (%f)
%	列26: 双精度值 (%f)
%   列27: 双精度值 (%f)
%	列28: 双精度值 (%f)
%   列29: 双精度值 (%f)
%	列30: 双精度值 (%f)
%   列31: 双精度值 (%f)
%	列32: 双精度值 (%f)
%   列33: 双精度值 (%f)
%	列34: 双精度值 (%f)
%   列35: 双精度值 (%f)
%	列36: 双精度值 (%f)
%   列37: 双精度值 (%f)
%	列38: 双精度值 (%f)
%   列39: 双精度值 (%f)
%	列40: 双精度值 (%f)
%   列41: 双精度值 (%f)
%	列42: 双精度值 (%f)
%   列43: 双精度值 (%f)
%	列44: 双精度值 (%f)
%   列45: 双精度值 (%f)
%	列46: 双精度值 (%f)
%   列47: 双精度值 (%f)
%	列48: 双精度值 (%f)
%   列49: 双精度值 (%f)
%	列50: 双精度值 (%f)
%   列51: 双精度值 (%f)
%	列52: 双精度值 (%f)
%   列53: 双精度值 (%f)
%	列54: 双精度值 (%f)
%   列55: 双精度值 (%f)
%	列56: 双精度值 (%f)
%   列57: 双精度值 (%f)
%	列58: 双精度值 (%f)
%   列59: 双精度值 (%f)
%	列60: 双精度值 (%f)
%   列61: 双精度值 (%f)
%	列62: 双精度值 (%f)
%   列63: 双精度值 (%f)
%	列64: 双精度值 (%f)
%   列65: 双精度值 (%f)
%	列66: 双精度值 (%f)
%   列67: 双精度值 (%f)
%	列68: 双精度值 (%f)
%   列69: 双精度值 (%f)
%	列70: 双精度值 (%f)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%s%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r');

%% 根据格式字符串读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'EmptyValue' ,NaN,'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'EmptyValue' ,NaN,'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% 关闭文本文件。
fclose(fileID);

%% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

%% 创建输出变量
dataArray([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]) = cellfun(@(x) num2cell(x), dataArray([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]), 'UniformOutput', false);
apascaltrain = [dataArray{1:end-1}];
