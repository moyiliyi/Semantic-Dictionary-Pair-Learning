function  [ ADictMat , AEncoderMat ] = dict_training(TrData,TrLabel,DictSize,AEncoderMat,ADictMat)
% Column normalization
TrData = normcol_equal(TrData);

%Parameter setting                

tau    = 0.05;
lambda = 0.003;
gamma  = 0.0001;

% DPL trainig
[ DictMat , EncoderMat ] = TrainDPL(  TrData, TrLabel, DictSize, tau, lambda, gamma );

end

