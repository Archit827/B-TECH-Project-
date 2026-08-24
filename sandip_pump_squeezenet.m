% --- SqueezeNet Transfer Learning for Air Compressor Fault Detection ---

doFeatureExtraction = true;
downloadFolder = fullfile('.','pump');
if ~exist(downloadFolder,'dir')
    error('Dataset not found. Please ensure pump folder exists.');
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
        data = mean(data, 2);
        spec = melSpectrogram(data,sampleRate, ...
            'WindowLength',windowLength, ...
            'OverlapLength',overlapLength, ...
            'FFTLength',fftLength, ...
            'NumBands',64);
        spec = log10(spec + eps); % Log compression
        trainingSpectrograms(:,:,:,index) = single(spec);
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
        data = mean(data, 2);
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

% ******************************************************************
% Expand single-channel spectrograms to 3 channels
trainSpecs = permute(trainingSpectrograms, [1 2 3 4]);
valSpecs = permute(validationSpectrograms, [1 2 3 4]);

trainSpecs = repmat(trainSpecs, [1 1 3 1]);
valSpecs = repmat(valSpecs, [1 1 3 1]);

trainLabels = adsTrain.Labels;
valLabels = adsValidation.Labels;

% Load Pretrained SqueezeNet
net = squeezenet;

% Modify the network for 8-class classification
lgraph = layerGraph(net);

% Replace final convolution and classification layers
numClasses = numel(categories(trainLabels));
newConvLayer = convolution2dLayer(1,numClasses,'Name','new_conv',...
    'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
newClassificationLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificationLayer);

% Resize images to match SqueezeNet input size
inputSize = net.Layers(1).InputSize; % Typically [227 227 3]
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
disp("Validation Accuracy with SqueezeNet: " + accuracy*100 + "%")

% Plot Confusion Matrix
figure;
confusionchart(valLabels, validationResults);
title('Confusion Matrix - SqueezeNet Transfer Learning')
save('trainedSqueezeNetModel.mat','airCompNet')

disp('Thanks ---- Sandip Kumar Lahiri')
