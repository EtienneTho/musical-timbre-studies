% compute the log attack time
% LAT : log attack time
% AT : attack time

function [LAT,AT] = LOGAttackTime(filename,thrMin,thrMax)

    [wavtemp, fs] = audioread(filename) ; % read file
    wavtemp = sqrt(wavtemp(1:end).^2) ; % abolute value
    wavtemp = wavtemp / max(wavtemp(1:floor(length(wavtemp)/3))) ;

    env = wavtemp ;
    ex = 1 ;
    idx = 1 ;
    t1 = 0;
    t2 = 0;
    while ex
        if env(idx) > thrMin && t1==0
            t1 = idx ;
        end
        if env(idx) > thrMax && t2==0
            t2 = idx ;
            ex = 0;
        end
        idx = idx + 1 ;
    end
    AT = (t2-t1)/fs;
    LAT = log10(AT) ;
end