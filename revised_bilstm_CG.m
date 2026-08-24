doFeatureExtraction = true; % set to true if extracting features again

downloadFolder = fullfile('.','pump');
if ~exist(downloadFolder,'dir')
    url = 'https://www.mathworks.com/supportfiles/audio/pump/pump.zip';
    loc = websave(downloadFolder,url);
    unzip(loc,downloadFolder)
end

dataFolder = downloadFolder;
ads = audioDatastore(dataFolder,'IncludeSubfolders',true,'LabelSource','foldernames');

rng(3); % for reproducibility
ads = shuffle(ads);

[adsTrain,adsValidation] = splitEachLabel(ads,0.9,0.1);
disp(countEachLabel(adsTrain))
disp(countEachLabel(adsValidation))

sampleData = read(adsTrain);
sampleDataCategory = adsTrain.Labels(1);

% Feature extraction setup
trainingFeatures = cell(1,numel(adsTrain.Files));
windowLength = 512;
overlapLength = 0;

aFE = audioFeatureExtractor('SampleRate',16e3, ...
    'Window',hamming(windowLength,'periodic'), ...
    'OverlapLength',overlapLength, ...
    'spectralCentroid',true, ...
    'spectralCrest',true, ...
    'spectralDecrease',true, ...
    'spectralEntropy',true, ...
    'spectralFlatness',true, ...
    'spectralKurtosis',true, ...
    'spectralRolloffPoint',true, ...
    'spectralSkewness',true, ...
    'spectralSlope',true, ...
    'spectralSpread',true);

if doFeatureExtraction
    reset(adsTrain);
    index = 1;
    tic;
    while hasdata(adsTrain)
        data = read(adsTrain);
        data = mean(data, 2);
        trainingFeatures{index} = extract(aFE,data);
        index = index + 1;
    end
    fprintf('Extraction took %f seconds.\n', toc);
    save("TrainingFeatures.mat", "trainingFeatures");
else
    load("TrainingFeatures.mat"); 
    disp("Training data features loaded.")
end

% Normalize training features
allTrainingFeatures = cat(1, trainingFeatures{:});
M = mean(allTrainingFeatures);
S = std(allTrainingFeatures);

for index = 1:numel(trainingFeatures)
    trainingFeatures{index} = ((trainingFeatures{index} - M) ./ S).';
end

% Generate and normalize validation features
validationFeatures = cell(1,numel(adsValidation.Files));

if doFeatureExtraction
    reset(adsValidation);
    index = 1;
    tic;
    while hasdata(adsValidation)
        data = read(adsValidation);
        data = mean(data, 2);
        validationFeatures{index} = extract(aFE,data);
        validationFeatures{index} = ((validationFeatures{index} - M) ./ S).';
        index = index + 1;
    end
    fprintf('Validation Extraction took %f seconds.\n', toc);
    save("ValidationFeatures.mat", "validationFeatures");
else
    load("ValidationFeatures.mat");
end

% Training and evaluation
doTraining = true;
doTesting = true;

% Dynamically determine number of classes
numClasses = numel(unique(adsTrain.Labels));

% Define network
layers = [ 
    sequenceInputLayer(size(trainingFeatures{1},1))
    bilstmLayer(300, "OutputMode", "sequence")
    dropoutLayer(0.4)
    bilstmLayer(300, "OutputMode", "sequence")
    dropoutLayer(0.4)
    bilstmLayer(300, "OutputMode", "last")
    fullyConnectedLayer(numClasses)   % dynamically set
    softmaxLayer
    classificationLayer
];

miniBatchSize = 16;
validationFrequency = floor(numel(trainingFeatures)/miniBatchSize);

options = trainingOptions("adam", ...
    "MaxEpochs",120, ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.05, ...
    "LearnRateDropPeriod",25,...
    'ValidationData',{validationFeatures,adsValidation.Labels}, ...
    'ValidationFrequency',validationFrequency);

% Train network
if doTraining
    airCompNet = trainNetwork(trainingFeatures,adsTrain.Labels,layers,options); 
    save("TrainedModel.mat", "airCompNet");
else
    load("TrainedModel.mat");
end

% Test network
if doTesting
    validationResults = classify(airCompNet, validationFeatures);
    save("ValidationResults.mat", "validationResults");
else
    load("ValidationResults.mat");
end

% Show confusion chart and accuracy
cm = confusionchart(validationResults, adsValidation.Labels);
accuracy = sum(validationResults == adsValidation.Labels) / numel(validationResults);
disp("Accuracy: " + accuracy * 100 + "%")

disp('Thanks---- Sandip Kumar Lahiri');
