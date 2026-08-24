% --- ResNet18 Transfer Learning for Air Compressor Fault Detection ---

doFeatureExtraction = true;
downloadFolder = fullfile('.','AirCompressorDataSet');
if ~exist(downloadFolder,'dir')
    error('Dataset not found. Please ensure AirCompressorDataSet folder exists.');
end
dataFolder = downloadFolder;
ads = audioDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames');

% Shuffle the dataset
rng(3);
ads = shuffle(ads);

% Split into training and validation sets
[adsTrain, adsValidation] = splitEachLabel(ads,0.9,0.1);
countEachLabel(adsTrain)
countEachLabel(adsValidation)

% Data Preparation: Create Spectrograms
windowLength = 512;
overlapLength = round(0.75 * windowLength);
fftLength = 512;
sampleRate = 16000;

% Prepare Training Spectrograms
if doFeatureExtraction
    reset(adsTrain);
    trainingSpectrograms = [];
    index = 1;
    tic;
    while hasdata(adsTrain)
        data = read(adsTrain);
        spec = melSpectrogram(data,sampleRate, ...
            'WindowLength',windowLength, ...
            'OverlapLength',overlapLength, ...
            'FFTLength',fftLength, ...
            'NumBands',64);
        spec = log10(spec + eps); % Log compress
        trainingSpectrograms(:,:,:,index) = single(spec); % Stack as 3D array
        index = index + 1;
    end
    fprintf('Training Spectrogram Extraction took %f seconds.\n',toc);
else
    load("TrainingSpectrograms.mat");
end

% Prepare Validation Spectrograms
if doFeatureExtraction
    reset(adsValidation);
    validationSpectrograms = [];
    index = 1;
    tic;
    while hasdata(adsValidation)
        data = read(adsValidation);
        spec = melSpectrogram(data,sampleRate, ...
            'WindowLength',windowLength, ...
            'OverlapLength',overlapLength, ...
            'FFTLength',fftLength, ...
            'NumBands',64);
        spec = log10(spec + eps);
        validationSpectrograms(:,:,:,index) = single(spec);
        index = index + 1;
    end
    fprintf('Validation Spectrogram Extraction took %f seconds.\n',toc);
else
    load("ValidationSpectrograms.mat");
end

% Make sure data is formatted as HxWxCxN
trainSpecs = permute(trainingSpectrograms, [1 2 3 4]);
valSpecs = permute(validationSpectrograms, [1 2 3 4]);

% Get labels
trainLabels = adsTrain.Labels;
valLabels = adsValidation.Labels;

% ******************************************************************
% Load Pretrained ResNet18
net = resnet18;

% Modify the network for our 8-class compressor classification
lgraph = layerGraph(net);

% Replace final layers
numClasses = numel(categories(trainLabels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fcNew')
    softmaxLayer('Name','softmaxNew')
    classificationLayer('Name','classificationNew')
    ];

% Remove original final layers
lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
lgraph = replaceLayer(lgraph,'prob',newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newLayers(3));

% Update input size if necessary
inputSize = net.Layers(1).InputSize;

% If spectrogram size is different from ResNet input size (224x224), resize:
augTrainSpecs = augmentedImageDatastore(inputSize(1:2), trainSpecs, trainLabels);
augValSpecs = augmentedImageDatastore(inputSize(1:2), valSpecs, valLabels);

% ******************************************************************
% Training Options
miniBatchSize = 32;
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValSpecs, ...
    'ValidationFrequency',floor(numel(trainLabels)/miniBatchSize), ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');

% ******************************************************************
% Train the Network
airCompNet = trainNetwork(augTrainSpecs, lgraph, options);

% ******************************************************************
% Evaluate the Network
validationResults = classify(airCompNet, augValSpecs);
accuracy = sum(validationResults == valLabels) / numel(valLabels);
disp("Validation Accuracy with ResNet18: " + accuracy*100 + "%")

% Confusion Matrix
figure;
confusionchart(valLabels, validationResults);
title('Confusion Matrix - ResNet18 Transfer Learning')

disp('Thanks ---- Sandip Kumar Lahiri')

