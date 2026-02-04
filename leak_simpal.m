 
 clc;close all;clear;%delete(findall(0));
digitDatasetPath = fullfile('I:\NEW RESERCHE\eye\test');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');
labelCount = countEachLabel(imdsTrain);
labelCount = countEachLabel(imdsValidation);

    layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)  

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)  
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(layers)
% ylim([0,10])


% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',10, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',50, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% options = trainingOptions('adam', ...
%     'MiniBatchSize',32, ...
%     'ExecutionEnvironment','gpu','MaxEpochs',20, ...
%     'ValidationData',imdsValidation, ...
%     'Verbose',1, ...
%     'Plots','training-progress');
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'ExecutionEnvironment','gpu','MaxEpochs',20, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'Verbose',1, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);
% [YValPred,probs] = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% 
% accuracy = mean(YValPred == YValidation)
% 
% scores = predict(net,imdsValidation);
% 
% 
% disp("Training error: " + accuracy*100 + "%")
% 
% figure(Units="normalized",Position=[0.2 0.2 0.4 0.4]);
% cm = confusionchart(YValidation,YValPred);
% cm.Title = "Confusion Matrix for Validation Data";
% cm.ColumnSummary = "column-normalized";
% cm.RowSummary = "row-normalized";
[YtrainPred,prob] = classify(net,imdsTrain);
Ytrain = imdsTrain.Labels;
trainaccuracy = mean(YtrainPred == Ytrain)
trainscores = predict(net,imdsValidation)
disp("Training-Accuracy: " + trainaccuracy*100 + "%")
figure(Units="normalized",Position=[0.2 0.2 0.4 0.4]);
cm = confusionchart(Ytrain,YtrainPred);
cm.Title = "Confusion Matrix for Training Data";
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";

% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
[YValPred,probs] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YValPred == YValidation)
scores = predict(net,imdsValidation);
disp("Training error: " + accuracy*100 + "%")
figure(Units="normalized",Position=[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YValPred);
cm.Title = "Confusion Matrix for Validation Data";
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";
