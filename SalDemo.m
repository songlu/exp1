clear
clc

%% addpath to SaliencyToolbox
addpath('../SaliencyToolbox/');
% setup saliency
sal_setup;

%% Parameters
Params.PATCH_SZ = 4;
Params.IPatchScales = [1, .8, .5, .3];
% NN patches are searched over current patch scale IPatchScales(i).*INeighbourMult
Params.INeighbourMult = [1, 0.5, .25];
Params.MaxSize = 250;
p_params.useApprox = 1;
p_params.UniquenessType = 1;
p_params.NN = 640;
Params.p_params = p_params;

% contains links to unlabelled dataset of images and precomputed GIST and
% LAB features for the dataset
NNIParams.DB{1} = '../../NNImgFeat/VOC2007/GISTLab/';       % feature files here
NNIParams.ImgLocDB{1} = '../../Data/VOC2007/JPEGImages/';   % images here
NNIParams.K = 20;
Params.NNIParams = NNIParams;

%% setup all the unlabled datasets (folder) if they haven't been already
for i = 1:numel(NNIParams.DB)
    SetupDataset(NNIParams.ImgLocDB{i}, NNIParams.DB{i});
end

%% Get saliancy of image

imageID = '007154';
I = imread('007154.jpg');

[gist1, GISTParam, lab] = GetGISTAndLab(I);

[S, SWI, SAI] = GetSalImg(I, imageID, gist1, lab, Params);

imwrite(S,'007154_S.png');
imwrite(SWI,'007154_SW.png');
imwrite(SAI,'007154_SA.png');

%% Get Bounding Boxes

ParamsBB.Num2Select = 100;
[selectedBB, AllBB] = SampleBBNMS_multiScale(S, ParamsBB);

%% Get Coherent sampling Bounding Boxes
[SIFTBoW] = GetBoWIDs(I);
BoWParams.Params.numWords = 2000;
shiftedBB = GetConsistantBoxes(AllBB, selectedBB, SIFTBoW, BoWParams);

%% Plot results
figure;
imshow(I)
hold on;
load tmp
box = selectedBB(1:4,1);
plotBox = [box(1) box(2); ...
box(1) box(4); ...
box(3) box(4); ...
box(3) box(2); ...
box(1) box(2)];
plot(plotBox(:,1),plotBox(:,2),'r');
box = shiftedBB(1:4,1);
plotBox = [box(1) box(2); ...
box(1) box(4); ...
box(3) box(4); ...
box(3) box(2); ...
box(1) box(2)];
plot(plotBox(:,1),plotBox(:,2));
legend('NMS Sampling', 'Coherent Sampling');