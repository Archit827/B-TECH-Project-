doFeatureExtraction = true; % if unchecked, this will save time by loading previous results
%url = 'https://www.mathworks.com/supportfiles/audio/AirCompressorDataset/AirCompressorDataset.zip';
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
% Split the data into training and validation by doing a 90% training, 10% validation split. The countEachLabel command will show us how many samples of data belong to each category in the dataset.
[adsTrain,adsValidation] = splitEachLabel(ads,0.9,0.1);
countEachLabel(adsTrain)
countEachLabel(adsValidation)

% ***************************Data Preparation
% Human Insight
% The data we are working with are time-series recordings of acoustics from different parts of an air compressor. As such, there are strong relationships between samples in time.
sampleData = read(adsTrain);
sampleDataCategory = adsTrain.Labels(1);
% plot(1:numel(sampleData), sampleData);
% xlabel("Sample number");
% ylabel("Amplitude");
% title("Class: " + string(sampleDataCategory));
% Listen to a sample of the audio if desired.
 %sound(sampleData,16000);
 
 
 %****************************************
%  Generate Training Features
% The next step is to extract the set of acoustic features that will be used as inputs to the network. 
% The Audio Toolbox provides a set of Spectral Descriptor features that are commonly used as inputs to deep learning networks.
% 
% We can extract the features with individual functions, or we can simplify the workflow and use a single object called audioFeatureExtractor to do it all at once. 
trainingFeatures = cell(1,numel(adsTrain.Files));
windowLength = 512;
overlapLength = 0;

aFE = audioFeatureExtractor('SampleRate',16e3, ...
    'Window',hamming(windowLength,'periodic'),...
    'OverlapLength',overlapLength,...
    'spectralCentroid',true, ...
    'spectralCrest',true, ...
    'spectralDecrease',true, ...
    'spectralEntropy',true,...
    'spectralFlatness',true,...
    'spectralFlux',false,...                
    'spectralKurtosis',true,...
    'spectralRolloffPoint',true,...
    'spectralSkewness',true,...
    'spectralSlope',true,...
    'spectralSpread',true);

if doFeatureExtraction
    reset(adsTrain);
    index = 1;
    tic;
    while hasdata(adsTrain)
        data = read(adsTrain);
        trainingFeatures{index} = extract(aFE,data);
        index = index + 1;
    end
    fprintf('Extraction took %f seconds.\n',toc);
else
    load("TrainingFeatures.mat"); 
    disp("Training data features loaded.")
end

% Normalize Training Features
% Networks will often train better when normalized. Calculate the mean and standard deviation and normalize each element of the training feature set. 
allTrainingFeatures = cat(1,trainingFeatures{:});
M = mean(allTrainingFeatures);
S = std(allTrainingFeatures);

for index = 1:numel(adsTrain.Files)
   trainingFeatures{index} =  ((trainingFeatures{index} - M)./S).';
end


% Generate and Normalize Validation Features
% Repeat the feature extraction for the validation features. Perform the normalization inside the loop.
validationFeatures = cell(1,numel(adsValidation.Files));

if doFeatureExtraction
    index = 1;
    tic;
    while hasdata(adsValidation)
       data = read(adsValidation);
       validationFeatures{index} = extract(aFE,data);
       validationFeatures{index} = ((validationFeatures{index}  - M) ./ S).';
       index = index + 1;
    end
    fprintf('Validation Extraction took %f seconds.\n',toc);
else
    load("ValidationFeatures.mat"); %#ok<*UNRCH> 
end



%*******************************************************************

% Air Compressor Data Classification
% Part 2: Train and Evaluate a Model



% Configuration
% Click on the checkboxes below to choose options for how to run this script.
doTraining = true;
doTesting = true;



% Load Data
% Load data that was preprocessed in the previous section.
load("TrainingFeatures.mat");
load("ValidationFeatures.mat");
%reloadDatastore;


%Use an LSTM network. An LSTM layer learns long-term dependencies between time steps of time series or sequence data. The first lstmlayer will have 100 hidden units and output the sequence data. Then a dropout layer will be used to reduce probability of overfitting. The second lstmlayer will output just the last step of the time sequence.
layers = [ 
    sequenceInputLayer(size(trainingFeatures{1},1))
    
    bilstmLayer(150, "OutputMode", "sequence")
    dropoutLayer(0.2)
    
    bilstmLayer(150, "OutputMode", "last")
    
    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer
];


%Define Network Hyperparameters
miniBatchSize = 32;
validationFrequency = floor(numel(trainingFeatures)/miniBatchSize);
options = trainingOptions("adam", ...
    "MaxEpochs",50, ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",20,...
    'ValidationData',{validationFeatures,adsValidation.Labels}, ...
    'ValidationFrequency',validationFrequency);

%************************************************************************

% Train The Network
% This network takes about 100 seconds to train on an NVIDIA RTX 2080 GPU.
if doTraining
    airCompNet = trainNetwork(trainingFeatures,adsTrain.Labels,layers,options); 
else
    load("TrainedModel.mat");
end

% Test The Network
% Now that the network has been trained, we can test it on the validation data. 
if doTesting
    validationResults = classify(airCompNet,validationFeatures);
else
    load("ValidationResults.mat");
end
% View the confusion chart for the test results:
cm = confusionchart(validationResults,adsValidation.Labels);
% View the overall accuracy percentage of the validation and test results:
accuracy = sum(validationResults == adsValidation.Labels) / numel(validationResults);
disp("Accuracy: " + accuracy * 100 + "%")

%%% This portion is optional

%Try out more network training in Experiment Manager (optional)
%experimentManager;

% Visualize LSTM Activations
% X = trainingFeatures{1};
% sequenceLength = size(X,2);
% idxLayer = 2;
% features = zeros(100,sequenceLength);
% 
% if doTesting
%     for i = 1:sequenceLength
%         features(:,i) = cell2mat(activations(airCompNet,X(:,i),idxLayer));  
%         [net, YPred(i)] = classifyAndUpdateState(airCompNet,X(:,i)); %#ok<SAGROW> 
%     end
% else
%     load("LSTMActivations.mat");
% end
% 
% %Visualize the first 40 hidden units using a heatmap.
% heatmap(features(1:40,1:40));
% xlabel("Time Step")
% ylabel("Hidden Unit")
% title("LSTM Activations")



%  end of program
disp('Thanks---- Sandip Kumar lahiri')