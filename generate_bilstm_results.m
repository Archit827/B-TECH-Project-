% --- Create results folder ---
resultsFolder = 'sandipresults_bilstm';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

% --- Save Confusion Matrix Figure ---
figure;
cm = confusionchart(adsValidation.Labels, validationResults);
cm.Title = 'Confusion Matrix - BiLSTM Model';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf, fullfile(resultsFolder, 'confusion_matrix_BiLSTM.png'));

% --- Also Save Confusion Matrix Values to CSV ---
confMat = confusionmat(adsValidation.Labels, validationResults);
classNames = categories(adsValidation.Labels);
confMatTable = array2table(confMat, 'VariableNames', classNames, 'RowNames', classNames);
writetable(confMatTable, fullfile(resultsFolder, 'confusion_matrix_BiLSTM.csv'), 'WriteRowNames', true);

% --- Compute per-class metrics ---
trueLabels = adsValidation.Labels;
predLabels = validationResults;
numClasses = numel(classNames);

precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1score = zeros(numClasses,1);
support = zeros(numClasses,1);

for i = 1:numClasses
    class = classNames{i};
    tp = sum((predLabels == class) & (trueLabels == class));
    fp = sum((predLabels == class) & (trueLabels ~= class));
    fn = sum((predLabels ~= class) & (trueLabels == class));
    
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
    support(i) = sum(trueLabels == class);
end

macroF1 = mean(f1score);
overallAcc = sum(predLabels == trueLabels) / numel(trueLabels);

% --- Display metrics table ---
resultsTable = table(classNames, ...
    round(precision*100,1), ...
    round(recall*100,1), ...
    round(f1score*100,1), ...
    support, ...
    'VariableNames', {'FaultType','Precision','Recall','F1_Score','Support'});

disp(resultsTable);
fprintf('\nOverall Accuracy: %.2f%%\n', overallAcc*100);
fprintf('Macro-F1 Score: %.2f%%\n', macroF1*100);

% --- Export performance table to Excel ---
filenameExcel = fullfile(resultsFolder, 'BiLSTM_Performance_Metrics.xlsx');
writetable(resultsTable, filenameExcel);

% --- Append Overall Accuracy and Macro-F1 Score to the Excel ---
% Read back the file
T = readtable(filenameExcel);
% Add two new rows manually
T.FaultType{end+1} = 'Overall Accuracy';
T.Precision(end+1) = overallAcc * 100;
T.Recall(end+1) = NaN;
T.F1_Score(end+1) = NaN;
T.Support(end+1) = NaN;

T.FaultType{end+1} = 'Macro-F1 Score';
T.Precision(end+1) = macroF1 * 100;
T.Recall(end+1) = NaN;
T.F1_Score(end+1) = NaN;
T.Support(end+1) = NaN;

% Save the updated table
writetable(T, filenameExcel);
