%%% Figure 2 - comparison between meta-analysis and original values (run
%%% main script before)
%% LAT
% new = [0.9140    0.2287    0.1759    0.5916    0.5176    0.7291    0.8656    0.3940    0.3412] ;
vecFig2 = [1 6 7 8 9 10 11 12 13] ;
new = max(tabMeanMaxPaper(vecFig2,3:4)')  ;
ori = [-.95 .79 .75 -.62 -.7 -.94 .97 .97 .97].^2 ;

% barthet | 

[r,p] = corr(abs(ori)',abs(new)','type','spearman') ;
subplot(121)
scatter(abs(ori),abs(new),'k','filled') ;
title('Log-Attack Time');
hold on; plot((0:0.1:1),(0:0.1:1),'k');
xlabel({'Original explained variance';'Mdn=.88'})
ylabel({'Meta-analysis explained variance';'Mdn=.67'})
axis([0 1 0 1]) ;
axis square
box on
labels = {'B2010', 'I1993W','L2000C','L2000H',...
          'L2000P', 'McA1995', 'P2012A3',...
          'P2012DX4', 'P2012GD4'};
[h, ext] = labelpoints(ori + [0 0 0 0 0 0 0 0 -1] * .02, new + [0 0 0 0 0 0 1 0 -1] * .02, labels);

%% SC
% new = [0.9224 0.1145 0.1092 0.5598 0.3134 0.7311 0.2309 0.5475 0.5851 0.4156 0.3967] ;
vecFig2 = [1 4 5 6 7 8 9 10 11 12 13] ;
new = max(tabMeanMaxPaper(vecFig2,3:4)') ;
ori = [-.94 -.61 -.75 .71 .74 -.91 -.89 -.94 .62 .62 .62].^2 ;


[r,p] = corr(abs(new)',abs(ori)','type','spearman') ;
subplot(122) ;
scatter(abs(ori),abs(new),'k','filled') ;
hold on; plot((0:0.1:1),(0:0.1:1),'k') ;

axis([0 1 0 1]) ;
title('Spectral Centroid');
xlabel({'Original explained variance';'Mdn=.54'})
ylabel({'Meta-analysis explained variance';'Mdn=.59'})
axis square
box on

labels = {'B2010', 'I1993O', 'I1993R', 'I1993W','L2000C','L2000H',...
          'L2000P', 'McA1995', 'P2012A3',...
          'P2012DX4', 'P2012GD4'};
[h, ext] = labelpoints(ori + [0 0 0 0 0 0 0 0 0 0 -1] * .02, new + [0 0 +1 0 0 -3 0 1 1 0 0] * .02, labels);

