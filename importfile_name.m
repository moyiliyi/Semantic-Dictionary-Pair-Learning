function attributenames = importfile_name(filename, startRow, endRow)
%IMPORTFILE1 将文本文件中的数值数据作为矩阵导入。
%   ATTRIBUTENAMES = IMPORTFILE1(FILENAME) 读取文本文件 FILENAME 中默认选定范围的数据。
%
%   ATTRIBUTENAMES = IMPORTFILE1(FILENAME, STARTROW, ENDROW) 读取文本文件
%   FILENAME 的 STARTROW 行到 ENDROW 行中的数据。
%
% Example:
%   attributenames = importfile1('attribute_names.txt', 1, 64);
%
%    另请参阅 TEXTSCAN。

% 由 MATLAB 自动生成于 2015/11/05 15:33:19

%% 初始化变量。
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% 每个文本行的格式字符串:
%   列1: 文本 (%s)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%s%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r');

%% 根据格式字符串读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    dataArray{1} = [dataArray{1};dataArrayBlock{1}];
end

%% 关闭文本文件。
fclose(fileID);

%% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

%% 创建输出变量
attributenames = [dataArray{1:end-1}];
