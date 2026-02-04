clc;close all;clear;%delete(findall(0));

digitDatasetPath = fullfile('C:\Users\Ay\OneDrive\Desktop\7-DEEP LEARNING APPROACH FOR CLASSIFICATION OF GLAUCOMA USING COLOR FUNDUS IMAGES\archive');

imds = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');


% numTrainingFiles = 0.8;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize');

net=resnet50;
analyzeNetwork(net)

net.Layers(1)
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 8;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'ExecutionEnvironment','auto','MaxEpochs',20, ...%'auto', 'gpu', 'cpu', 'multi-gpu', 'parallel'
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YValPred,probs] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YValPred == YValidation)

scores = predict(net,imdsValidation);


disp("Training error: " + accuracy*100 + "%")

figure(Units="normalized",Position=[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YValPred);
cm.Title = "Confusion Matrix for Validation Data";
% cm.ColumnSummary = "column-normalized";
% cm.RowSummary = "row-normalized";




