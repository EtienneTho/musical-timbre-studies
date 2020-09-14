function [resultsStruct] = CorrelationOfDescriptorsWithMdsDimensions(soundPath, ext , dissimilaritiesFile, AT, nbDimMDS, typeOfCorrelation,rotation)

grey1977 = [-0.276	0.199;
0.262	-0.226;
-0.096	-0.326;
0.19	0.056;
-0.401	0.064;
-0.209	0.072;
0.203	-0.048;
0.291	0.118;
-0.068	0.208;
-0.249	0.146;
-0.435	-0.073;
0.433	0.254;
-0.067	0.29;
0.338	-0.284;
-0.014	-0.288;
0.099	-0.162];

grey1978 = [-0.424	-0.261;
0.332	0.243;
0.158	0.267;
0.122	-0.282;
-0.364	-0.308;
-0.211	0.176;
0.025	0.227;
0.269	-0.203;
-0.178	0.172;
0.135	0.271;
-0.471	0.109;
0.149	-0.17;
0.23	-0.219;
0.221	0.024;
-0.065	0.134;
0.072	-0.18];

mcadams1995 = [-3.35	1.34;
-2.58	-1.91;
-2.42	1.73;
2.98	1.72;
-8.40E-02	-2.69;
2.98	1.72;
3.78	1.77;
-1.39	-0.87;
3.57	-2.76;
-1.91	-1.5;
-2.39	-1.85;
-2.36	1.95;
0.73	2.29;
2.53	-2.27;
2.9	0.21;
-2.45	-1.36;
1.29	1.3;
-1.81	1.19];

% Attack Time Parameters
thrMin = 10^(AT.thrMin/20) ; % db to amp
thrMax = 10^(AT.thrMax/20) ; % db to amp

% initialize sound path
addpath(genpath('timbreToolbox')) ;
addpath(soundPath) ;
soundsList = dir(strcat(soundPath, '*.',ext)) ;
nbSounds = length(soundsList) ;
sc = zeros(nbSounds,1) ; % spectral centroid tab
tabDescriptors = struct('DescStat',{}) ;

% compute timbre descriptors : log attack time with arguments and Spectral Centroid
for iFile = 1:nbSounds % loop on the sounds
    filename  = strcat(soundPath,soundsList(iFile).name) ;        
    sc(iFile) = spectralCentroid(filename) ;
    [tabDescriptors(iFile).LAT,tabDescriptors(iFile).AT] = LOGAttackTime(filename,thrMin,thrMax) ;       
end

tab_SpectralCentroid = zeros(nbSounds,1) ;
tab_LogAttackTime    = zeros(nbSounds,1) ;
dissimilarities      = load(dissimilaritiesFile) ;

%% sort descriptors
for iSound = 1:nbSounds
    tab_LogAttackTime(iSound)    = log10(tabDescriptors(iSound).AT) ;    
    tab_SpectralCentroid(iSound) = sc(iSound);  
end

descriptors = [tab_LogAttackTime tab_SpectralCentroid];
          
%% compute MDS
% MDS on averaged dissimilarity matrix across subjects
[MDS,stress]      = mdscale((dissimilarities+dissimilarities')/2,nbDimMDS) ; 


% corr function
mycorr = @(x1,x2) corr(x1,x2,'type',typeOfCorrelation).^2 ;
mycorr_sqrt = @(x1,x2) corr(x1,x2,'type',typeOfCorrelation) ;

nIterations = 2000;

% Rotation of the MDS
if rotation == 0
    descriptors = [descriptors zeros(length(MDS(:,1)),length(MDS(1,:))-2)] ;
    [~,MDS_rot] = procrustes(descriptors,MDS) ;
    corrTabMDSDesc = zeros(2,2,1) ; % correlation of each dimensions with each descriptors
    CI95 = zeros(2,2,1) ; % CI95 of correlation of each dimensions with each descriptors
    [corrTabMDSDesc(1,1,1),corrTabMDSDesc(1,1,2)] = corr(descriptors(:,1),MDS_rot(:,1),'type',typeOfCorrelation) ;
    tmp_ = bootci(nIterations,{mycorr,descriptors(:,1),MDS_rot(:,1)})' ;
    CI95(1,1,1) = tmp_(1) ; 
    CI95(1,1,2) = tmp_(2) ;
    [corrTabMDSDesc(2,2,1),corrTabMDSDesc(2,2,2)] = corr(descriptors(:,2),MDS_rot(:,2),'type',typeOfCorrelation) ;
    tmp_ = bootci(nIterations,{mycorr,descriptors(:,2),MDS_rot(:,2)})' ;
    CI95(2,2,1) = tmp_(1) ; 
    CI95(2,2,2) = tmp_(2) ;
    
    MDS = [MDS(:,2) MDS(:,1)] ;
    [~,MDS_rot] = procrustes(descriptors,MDS) ;
    [corrTabMDSDesc(1,2,1),corrTabMDSDesc(1,2,2)] = corr(descriptors(:,1),MDS_rot(:,2),'type',typeOfCorrelation) ;
    tmp_ = bootci(nIterations,{mycorr,descriptors(:,1),MDS_rot(:,2)})' ;
    CI95(1,2,1) = tmp_(1) ; 
    CI95(1,2,2) = tmp_(2) ;
    [corrTabMDSDesc(2,1,1),corrTabMDSDesc(2,1,2)] = corr(descriptors(:,2),MDS_rot(:,1),'type',typeOfCorrelation) ;
    tmp_ = bootci(nIterations,{mycorr,descriptors(:,2),MDS_rot(:,1)})' ;
    CI95(2,1,1) = tmp_(1) ; 
    CI95(2,1,2) = tmp_(2) ;
    
%     corr(descriptors(:,1),grey1978(:,1),'type',typeOfCorrelation)
%     corr(descriptors(:,1),grey1978(:,2),'type',typeOfCorrelation)
%     corr(descriptors(:,2),grey1978(:,1),'type',typeOfCorrelation)
%     corr(descriptors(:,2),grey1978(:,2),'type',typeOfCorrelation)
%     pause    
else
    corrTabMDSDesc = zeros(2,2,1) ; % correlation of each dimensions with each descriptors
    [corrTabMDSDesc(1,1,1),corrTabMDSDesc(1,1,2)] = corr(descriptors(:,1),MDS(:,1),'type',typeOfCorrelation) ;
    [corrTabMDSDesc(1,2,1),corrTabMDSDesc(1,2,2)] = corr(descriptors(:,1),MDS(:,2),'type',typeOfCorrelation) ;
    [corrTabMDSDesc(2,1,1),corrTabMDSDesc(2,1,2)] = corr(descriptors(:,2),MDS(:,1),'type',typeOfCorrelation) ;
    [corrTabMDSDesc(2,2,1),corrTabMDSDesc(2,2,2)] = corr(descriptors(:,2),MDS(:,2),'type',typeOfCorrelation) ;
end

descriptors = [descriptors(:,1) descriptors(:,2)] ;

                    
% compile results in the output resultsStruct structure
resultsStruct.corrTabMDSDesc   = corrTabMDSDesc ;
resultsStruct.CI95   = CI95 ;
resultsStruct.mdsStress = stress ;

end

