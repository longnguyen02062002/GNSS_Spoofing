function acqResults = acquisition (longSignal, settings)
run('runsec.m');
save_results = zeros(1, 5250);
%% Initialization
% Find number of samples per spreading code
samplesPerCode = round(settings.samplingFreq / (settings.codeFreqBasis / settings.codeLength));

% Create two 1msec vectors of data to correlate with and one with zero DC
signal1 = longSignal(1 : samplesPerCode);
signal2 = longSignal(samplesPerCode+1 : 2*samplesPerCode);

signal0DC = longSignal - mean(longSignal);

% Find sampling period
ts = 1 / settings.samplingFreq;

% Find phase points of the local carrier wave
phasePoints = (0 : (samplesPerCode-1)) * 2 * pi * ts;

% Number of the frequency bins for the given acquisition band (500Hz steps)
numberOfFrqBins = round(settings.acqSearchBand * 2) + 1;

% Generate all C/A codes and sample them according to the sampling freq
caCodesTable = makeCaTable(settings);


% Initialize arrays to speed up the code
% Search results of all frequency bins and code shifts (for one satellite)
results = zeros (numberOfFrqBins, samplesPerCode);

% Carrier frequencies of the frequency bins
frqBins = zeros (1, numberOfFrqBins);

% Initialize acqResults
% Carrier frequencies of the frequency bins
acqResults.carrFreq     = zeros (1, 32);

% C/A code phases of detected signals
acqResults.codePhase    = zeros (1, 32);

% Correllation peak ratios ò the detected signals
acqResults.peakMetric   = zeros (1, 32);

fprintf ('(');

% Perform search for all listed PRN numbers
for PRN = settings.acqSatelliteList
%% Correlate siganals
    % Perform DFT of C/A code
    caCodeFreqDom = conj(fft(caCodesTable(PRN, :)));

    % Make the correlation for whole frequency band (for all freq. bins)
    for frqBinIndex = 1:numberOfFrqBins

        % Generate carrier wave frequency grid (0.5kHz step)
        frqBins(frqBinIndex) = settings.IF - (settings.acqSearchBand/2) * 1000 + 0.5e3 * (frqBinIndex - 1);

        % Generate local sine and cosine
        sinCarr = sin(frqBins(frqBinIndex) * phasePoints);
        cosCarr = cos(frqBins(frqBinIndex) * phasePoints);

        % Remove carrier from the signal
        IQ1 = (sinCarr+1i*cosCarr).* signal1;
        I1  = real (IQ1);
        Q1  = imag (IQ1);

        IQ2 = (sinCarr+1i*cosCarr).* signal2;
        I2  = real (IQ2);
        Q2  = imag (IQ2);

        % Convert the baseband signal to frequency domain (correlation in time domain)
        IQfreqDom1 = fft (I1 + 1i*Q1);
        IQfreqDom2 = fft (I2 + 1i*Q2);

        % Multiplication in the frequency domain
        convCodeIQ1 = IQfreqDom1 .* caCodeFreqDom;
        convCodeIQ2 = IQfreqDom2 .* caCodeFreqDom;

        % Perform inverse DFT and store correlation results
        acqRes1 = abs (ifft(convCodeIQ1)) .^ 2;
        acqRes2 = abs (ifft(convCodeIQ2)) .^ 2;
        
        % Check which msec had the greater power and save that, will
        % "blend" 1st and 2nd msec but will correct data bit issues
        if (max(acqRes1) > max(acqRes2))
            results (frqBinIndex, :) = acqRes1;
        else
            results (frqBinIndex, :) = acqRes2;
        end
    end % frqBinIndex = 1:numberOfFrqBins
    
%% Look for correlation peaks in the results
    % Find the highest peak and compare it to the second highest peak
    % The second peak is chosen not closer than 1 chip to the highest peak

    % Find the correlation peak and the carrier frequency
    [peakSize frequencyBinIndex] = max(max(results, [], 2)); % Maximum value on each row

    % Find code phase of the same correlation peak
    [peakSize codePhase] = max (max(results));

    % Find 1 chip wide C/A code phase exclude range around the peak
    samplesPerCodeChip = round (settings.samplingFreq / settings.codeFreqBasis);
    excludeRangeIndex1 = codePhase - samplesPerCodeChip;
    excludeRangeIndex2 = codePhase + samplesPerCodeChip;

    % Correct C/A code phase exclude range if the range includes array boundaries
    if excludeRangeIndex1 < 2
        codePhaseRange = excludeRangeIndex2 : samplesPerCode;
    elseif excludeRangeIndex2 >= samplesPerCode
        codePhaseRange = (excludeRangeIndex2 - samplesPerCode) : excludeRangeIndex1;
    else
        codePhaseRange = [1:excludeRangeIndex1, excludeRangeIndex2 : samplesPerCode];
    end

    try
        secondPeakSize = max(results(frequencyBinIndex, codePhaseRange));
    catch exception
        msgbox (exception.message);
    end;

    % Store result
    acqResults.peakMetric(PRN) = peakSize/secondPeakSize;
    % If the result is above threshold, then there is a signal
    if (peakSize/secondPeakSize) > settings.acqThreshold
        freq_Range = 1:numberOfFrqBins;
        xR = settings.IF - (settings.acqSearchBand/2) * 1000 + 0.5e3 * (freq_Range - 1);
        yR = [0 : samplesPerCode-1];
        new_results=reshape(results.', 1, []);
        new_small_results=zeros(1, 5250);
        for count=1:5250
            start_idx=(count-1)*100+1;
            end_idx=count*100;
            new_small_results(count)=mean(new_results(start_idx:end_idx));
        end
        save_results(settings.index, :)=new_small_results;
        %figure;
        %surf(results);
        %shading interp, title ("PRN " + PRN);

        %savePath = 'E:\image';
        %filename = snd + "_PRN_" + PRN + "_" + datestr(now,'yyyymmdd_HHMMSS') + ".png";

        %saveas(gcf, fullfile(savePath, filename));
        settings.index=settings.index+1;

%% Fine resolution frequency search
        % Indicate PRN number of the detected signal
        fprintf ('%02d ', PRN);

        % Generate 10msec long C/A codes sequence for given PRN
        caCode = generateCAcode(PRN);
        codeValueIndex = floor ((ts * (1 : 10*samplesPerCode)) / (1/settings.codeFreqBasis));
        longCaCode = caCode (rem(codeValueIndex, 1023) + 1);

        % Remove C/A code modulation from the original signal
        % (Using detected C/A code phase)
        xCarrier = signal0DC (codePhase : (codePhase + 10 * samplesPerCode - 1)) .* longCaCode;

        % Find the next highest power of two and increase by 8x
        fftNumPts = 8 * (2^(nextpow2(length(xCarrier))));

        % Compute the magnitude of the FFT, find maximum and the associated carrier frequency
        fftxc = abs (fft(xCarrier, fftNumPts));
        uniqFftPts = ceil ((fftNumPts + 1) / 2);
        [fftMax, fftMaxIndex] = max (fftxc);
        fftFreqBins = (0 : fftNumPts) * settings.samplingFreq/fftNumPts;

        % Save properties of the detected satellite signal
        acqResults.carrFreq (PRN) = fftFreqBins (fftMaxIndex);
        acqResults.codePhase (PRN) = codePhase;

    else
        % No signal with this PRN
        fprintf ('. ');
    end % if (peakSize/secondPeakSize) > settings.acqThreshold

end % for PRN = satelliteList

filename=sprintf('results_%d.mat', snd);
save (filename, 'save_results', '-v7.3');
% Acquisition is over
fprintf(')\n');
