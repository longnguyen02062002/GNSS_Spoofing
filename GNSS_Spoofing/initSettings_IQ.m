function settings = initSettings_IQ();

%% Optional Parameters
% Number of milliseconds to be processed used 36000 + any transients
settings.msToProcess        = 5*1e3;    %[ms]

% Move the starting point of processing.
%run('runskip.m');
settings.skipNumberOfBytes  = 0*4*2.5e7;

% The notch bandwidth of filter
settings.Brej       = 5e3;
%settings.fileName   = 'D:\Long\NAVIS\Data\2023-12-20.bin';
settings.fileName = 'Z:\ds7.bin';

settings.relativeFreq = 0;

%% Independent parameters
% Number of channels to be used for signal processing
settings.numberOfChannels   = 5;

% Intermediate, sampling and code frequencies
settings.IF             = 0e3;      %[Hz]
settings.samplingFreq   = 2.5e7; %2e6; [Hz]
settings.codeFreqBasis  = 1.023e6;      %[Hz]

% Define number of chips in a spreading code
settings.codeLength     = 1023;
settings.samplesPDI     = settings.samplingFreq*10e-3;

% Number of samples per spreading code
settings.samplesPerCode = round(settings.samplingFreq / (settings.codeFreqBasis / settings.codeLength));
settings.dataType       = 'int16';
if strcmp(settings.dataType,'int8')
    settings.dataTypeSize = 1;
else
    settings.dataTypeSize = 2;
end;

%% Acquisition settings

% Index of the graph
settings.index=1;

% Skips acquisition in the script postProcessing.m if set to 1
settings.skipAcquisition = 2e6;

% List of satellites to look for. Some satellites can be excluded to speed up acquisition
settings.acqSatelliteList = 1:32; %PRN numbers

% Band around IF to search for satellite signal. Depends on max Doppler
settings.acqSearchBand = 10; %[kHz]

% Threshold for the signal presence decision rule
settings.acqThreshold = 2.4;

%% Plot settings
% Enable/disable plotting of the tracking results for each channel
settings.plotAcquisition = 1; %0 - Off, 1 - On

%% Constants
settings.c                  = 299792458;    % The speed of light, [m/s]
settings.startOffset        = 68.802;       %[ms] Initial sign. travel time
