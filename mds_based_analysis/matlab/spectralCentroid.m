% compute spectral centroid from FFT

function sc = spectralCentroid(filename)

    [wavtemp, fs] = audioread(filename) ; % read file
    
    nfft = 2^nextpow2(length(wavtemp)) ; % nfft points
    FFT_ = (abs(fft(wavtemp,nfft))) ; % FFT modulus
    FFT_ = (FFT_ / max(FFT_));
    %FFT_(FFT_<.1) = 0 ;
    f = fs*(0:(nfft/2))/nfft ; % frequency range    
    sc = sum(f' .* FFT_(1:end/2+1)) / sum(FFT_(1:end/2+1)) ; % compute spectral centroid
    
end