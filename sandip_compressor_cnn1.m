doFeatureExtraction = true; % if unchecked, this will save time by loading previous results 
downloadFolder = fullfile('.','AirCompressorDataSet');
if ~exist(fullfile('.','AirCompressorDataSet'),'dir')
    loc = websave(downloadFolder,url);
    unzip(loc,fullfile('.','AirCompressorDataSet'))
end
dataFolder = downloadFolder;
ads = audioDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames');

%************
rng(3);
ads = shuffle(ads);
%*****************************

% Split Into Training and Validation Sets
[adsTrain, adsValidation] = splitEachLabel(ads, 0.9, 0.1);
countEachLabel(adsTrain)
countEachLabel(adsValidation)

% *************************** Data Preparation
sampleData = read(adsTrain);
sampleDataCategory = adsTrain.Labels(1);

% ****************************************
% Generate Training Spectrograms
trainingSpectrograms = cell(1, numel(adsTrain.Files));
windowLength = 512;
overlapLength = round(0.75 * windowLength); % 75% overlap
fftLength = 512;
sampleRate = 16000;

if doFeatureExtraction
    reset(adsTrain);
    index = 1;
    tic;
    while hasdata(adsTrain)
        data = read(adsTrain);
        spec = melSpectrogram(data, sampleRate, ...
                'WindowLength', windowLength, ...
                'OverlapLength', overlapLength, ...
                'FFTLength', fftLength, ...
                'NumBands', 64);
        trainingSpectrograms{index} = log10(spec + eps); % Log-compressed
        index = index + 1;
    end
    fprintf('Training Spectrogram Extraction took %f seconds.\n', toc);
else
    load("TrainingSpectrograms.mat"); 
    disp("Training Spectrograms Loaded.")
end

% Normalize Training Spectrograms
allTrainingSpectrograms = cat(3, trainingSpectrograms{:});
M = mean(allTrainingSpectrograms(:));
S = std(allTrainingSpectrograms(:));

for index = 1:numel(adsTrain.Files)
   trainingSpectrograms{index} = (trainingSpectrograms{index} - M) / S;
end

% Generate and Normalize Validation Spectrograms
validationSpectrograms = cell(1, numel(adsValidation.Files));

if doFeatureExtraction
    reset(adsValidation);
    index = 1;
    tic;
    while hasdata(adsValidation)
       data = read(adsValidation);
       spec = melSpectrogram(data, sampleRate, ...
                'WindowLength', windowLength, ...
                'OverlapLength', overlapLength, ...
                'FFTLength', fftLength, ...
                'NumBands', 64);
       validationSpectrograms{index} = (log10(spec + eps) - M) / S;
       index = index + 1;
    end
    fprintf('Validation Spectrogram Extraction took %f seconds.\n', toc);
else
    load("ValidationSpectrograms.mat");
end

% *******************************************************************
% Convert to 4D Arrays (Fix)

% Training Data
numTrain = numel(trainingSpectrograms);
trainSpecs = zeros([size(trainingSpectrograms{1},1), size(trainingSpectrograms{1},2), 1, numTrain],'single');
for i = 1:numTrain
    trainSpecs(:,:,1,i) = single(trainingSpectrograms{i});
end

% Validation Data
numValidation = numel(validationSpectrograms);
valSpecs = zeros([size(validationSpectrograms{1},1), size(validationSpectrograms{1},2), 1, numValidation],'single');
for i = 1:numValidation
    valSpecs(:,:,1,i) = single(validationSpectrograms{i});
end

% *******************************************************************
% Air Compressor Data Classification (CNN Based)

doTraining = true;
doTesting = true;

% CNN Model Layers
inputSize = size(trainSpecs(:,:,:,1));
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(adsTrain.Labels)))
    softmaxLayer
    classificationLayer
];

% Define Network Hyperparameters
miniBatchSize = 32;
validationFrequency = floor(numTrain/miniBatchSize);
options = trainingOptions("adam", ...
    "MaxEpochs",50, ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",20,...
    'ValidationData',{valSpecs, adsValidation.Labels}, ...
    'ValidationFrequency',validationFrequency);

% *******************************************************************
% Train The Network
if doTraining
    airCompNet = trainNetwork(trainSpecs, adsTrain.Labels, layers, options); 
else
    load("TrainedCNNModel.mat");
end

% Test The Network
if doTesting
    validationResults = classify(airCompNet, valSpecs);
else
    load("ValidationResults.mat");
end

% View the confusion chart
cm = confusionchart(validationResults, adsValidation.Labels);
accuracy = sum(validationResults == adsValidation.Labels) / numel(validationResults);
disp("Accuracy: " + accuracy * 100 + "%")

disp('Thanks---- Sandip Kumar Lahiri')
