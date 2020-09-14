clearvars;
clc;

datasetNames  = dir('../../ext/matlab/sounds/') ; % load sounds
nbDatasets = length(datasetNames)-3 ; 

tabMDS_results = [] ;
tabMDS_p = [] ;
tabMDS_ci95 = [] ;

for iDim = 2:10 % loop on all the 8 dimensions
    nbDims = ones(1,20)*iDim ; 
    for iFolder = 4:20 % loop on the datasets
%     for iFolder = 6 % loop on the datasets
        AT.thrMin = -40 ; % attack threshold in dB
        AT.thrMax = -12 ; % attack threshold in dB
        ext = 'aiff' ; % audio files extension
        datasetName =  datasetNames(iFolder).name ; % dataset name
        disp(['Dataset name : ', datasetName,' - Nb MDS Dimensions : ',num2str(iDim)]);
        soundPath = strcat('../../ext/matlab/sounds/',datasetName,'/') ; % soundpath of soundfiles
        dissimilaritiesFile = sprintf('../../ext/matlab/data/%s_dissimilarity_matrix.txt',datasetName) ; % load dissimilarity matrix
        [resultsStruct{iFolder-3}] = CorrelationOfDescriptorsWithMdsDimensions(soundPath, ext, dissimilaritiesFile, AT, nbDims(iFolder-3),'spearman',0) ;
    end

    for iDataset = 1:17
        tabMDS_results(iDim-1,iDataset,1) = resultsStruct{iDataset}.corrTabMDSDesc(1,1,1) ;
        tabMDS_results(iDim-1,iDataset,2) = resultsStruct{iDataset}.corrTabMDSDesc(1,2,1) ;
        tabMDS_results(iDim-1,iDataset,3) = resultsStruct{iDataset}.corrTabMDSDesc(2,1,1) ;
        tabMDS_results(iDim-1,iDataset,4) = resultsStruct{iDataset}.corrTabMDSDesc(2,2,1) ;
        
        tabMDS_p(iDim-1,iDataset,1) = resultsStruct{iDataset}.corrTabMDSDesc(1,1,2) ;
        tabMDS_p(iDim-1,iDataset,2) = resultsStruct{iDataset}.corrTabMDSDesc(1,2,2) ;
        tabMDS_p(iDim-1,iDataset,3) = resultsStruct{iDataset}.corrTabMDSDesc(2,1,2) ;
        tabMDS_p(iDim-1,iDataset,4) = resultsStruct{iDataset}.corrTabMDSDesc(2,2,2) ;

        tabMDS_ci95(iDim-1,iDataset,1,1) = resultsStruct{iDataset}.CI95(1,1,1) ;
        tabMDS_ci95(iDim-1,iDataset,2,1) = resultsStruct{iDataset}.CI95(1,2,1) ;
        tabMDS_ci95(iDim-1,iDataset,3,1) = resultsStruct{iDataset}.CI95(2,1,1) ;
        tabMDS_ci95(iDim-1,iDataset,4,1) = resultsStruct{iDataset}.CI95(2,2,1) ;        

        tabMDS_ci95(iDim-1,iDataset,1,2) = resultsStruct{iDataset}.CI95(1,1,2) ;
        tabMDS_ci95(iDim-1,iDataset,2,2) = resultsStruct{iDataset}.CI95(1,2,2) ;
        tabMDS_ci95(iDim-1,iDataset,3,2) = resultsStruct{iDataset}.CI95(2,1,2) ;
        tabMDS_ci95(iDim-1,iDataset,4,2) = resultsStruct{iDataset}.CI95(2,2,2) ;
        
        tabMDS_results(iDim-1,iDataset,5) = resultsStruct{iDataset}.mdsStress ;
    end
end
% %% results reported in the paper
% 
% % the order in the table doesn't correspond to the one in the paper but to
% % the alphabetic order of the folders
% colMean1 = mean(tabMDS_results(:,:,1).^2,1)'; %% ==> averaged explaned variance between dimension 1 and LAT
% colMean2 = mean(tabMDS_results(:,:,2).^2,1)'; %% ==> averaged explaned variance between dimension 2 and LAT
% colMean3 = mean(tabMDS_results(:,:,3).^2,1)'; %% ==> averaged explaned variance between dimension 1 and spectral centroid
% colMean4 = mean(tabMDS_results(:,:,4).^2,1)'; %% ==> averaged explaned variance between dimension 2 and spectral centroid
% 
% colStd1 = std(tabMDS_results(:,:,1).^2,1)'; %% ==> std explaned variance between dimension 1 and LAT
% colStd2 = std(tabMDS_results(:,:,2).^2,1)'; %% ==> std explaned variance between dimension 2 and LAT
% colStd3 = std(tabMDS_results(:,:,3).^2,1)'; %% ==> std explaned variance between dimension 1 and spectral centroid
% colStd4 = std(tabMDS_results(:,:,4).^2,1)'; %% ==> std explaned variance between dimension 2 and spectral centroid
% 
% tabMean = [colMean1 colMean2 colMean3 colMean4] 
% mean(tabMean)


%% 

% the order in the table doesn't correspond to the one in the paper but to
% the alphabetic order of the folders
[colMean1,i1] = max(tabMDS_results(:,:,1).^2,[],1); %% ==> max averaged explaned variance between dimension 1 and LAT
[colMean2,i2] = max(tabMDS_results(:,:,2).^2,[],1); %% ==> max averaged explaned variance between dimension 2 and LAT
[colMean3,i3] = max(tabMDS_results(:,:,3).^2,[],1); %% ==> max averaged explaned variance between dimension 1 and spectral centroid
[colMean4,i4] = max(tabMDS_results(:,:,4).^2,[],1); %% ==> max averaged explaned variance between dimension 2 and spectral centroid

pMax1 = [];
pMax2 = [];
pMax3 = [];
pMax4 = [];

CI_lower_1 = [] ;
CI_lower_2 = [] ;
CI_lower_3 = [] ;
CI_lower_4 = [] ;

CI_upper_1 = [] ;
CI_upper_2 = [] ;
CI_upper_3 = [] ;
CI_upper_4 = [] ;

for iii = 1:17
    pMax1 = [pMax1 tabMDS_p(i1(iii),iii,1)] ;
    pMax2 = [pMax2 tabMDS_p(i2(iii),iii,2)] ;
    pMax3 = [pMax3 tabMDS_p(i3(iii),iii,3)] ;
    pMax4 = [pMax4 tabMDS_p(i4(iii),iii,4)] ;
    
    CI_lower_1 = [CI_lower_1 tabMDS_ci95(i1(iii),iii,1,1)] ;
    CI_lower_2 = [CI_lower_2 tabMDS_ci95(i1(iii),iii,2,1)] ;
    CI_lower_3 = [CI_lower_3 tabMDS_ci95(i1(iii),iii,3,1)] ;
    CI_lower_4 = [CI_lower_4 tabMDS_ci95(i1(iii),iii,4,1)] ;
    
    CI_upper_1 = [CI_upper_1 tabMDS_ci95(i1(iii),iii,1,2)] ;
    CI_upper_2 = [CI_upper_2 tabMDS_ci95(i1(iii),iii,2,2)] ;
    CI_upper_3 = [CI_upper_3 tabMDS_ci95(i1(iii),iii,3,2)] ;
    CI_upper_4 = [CI_upper_4 tabMDS_ci95(i1(iii),iii,4,2)] ;    
end

tabMeanMaxPaper = [colMean1' colMean2' colMean3' colMean4']
tabMeanMaxPaper_p = [pMax1' pMax2' pMax3' pMax4']
tabMeanMaxPaper_lower = [CI_lower_1' CI_lower_2' CI_lower_3' CI_lower_4']
tabMeanMaxPaper_upper = [CI_upper_1' CI_upper_2' CI_upper_3' CI_upper_4']

median(tabMeanMaxPaper)
iqr(tabMeanMaxPaper)

[median(max(tabMeanMaxPaper(:,1:2),[],2)) median(max(tabMeanMaxPaper(:,3:4),[],2))]
[iqr(max(tabMeanMaxPaper(:,1:2),[],2)) iqr(max(tabMeanMaxPaper(:,3:4),[],2))]


