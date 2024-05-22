%--- Include folders with functions ---------------------------------------
addpath include             % The software receiver functions
addpath geoFunctions        % Position calculation related functions

%% Clean up the environment first =========================================
format ('compact');
format ('long', 'g');
clc

%--- Include folders with functions ---------------------------------------
addpath include
fprintf('-------------------------------\n\n');

%% Initialize the setting
global settings;
settings = initSettings_IQ();
disp('  ');
fprintf('Probing data (%s)...\n', settings.fileName)
[fid, message]  = fopen(settings.fileName, 'rb');

%% Acquisition ============================================================
for x=150.01:0.01:200 % x là số giây muốn bỏ qua
    %run('runskip.m');
    settings.skipNumberOfBytes = x*4*settings.samplingFreq;
    fseek(fid, settings.skipNumberOfBytes, 'bof');
    tmp = fread(fid, 2*15*settings.samplesPerCode, settings.dataType)';
    data=tmp(1:2:end)+1i*tmp(2:2:end);
    %hist(tmp(1:2:end),20); grid on;
    %xlabel('Bin');
    %ylabel('Number in Bin');
    
    %--- Do the acquisition -------------------------------------------
    disp ('Acquiring satellites...');
    acqResults = acquisitionIQ(data, settings);
    %plotAcquisition(acqResults);
    
    settings.numberOfChannels = min([settings.numberOfChannels, sum(acqResults.peakMetric > settings.acqThreshold)]);
    if (any(acqResults.peakMetric > settings.acqThreshold))
        channel = preRun(acqResults, settings);
        showChannelStatus(channel, settings);
    else
        % No satellites to track, exit
        disp('No GNSS signals detected, signal processing finished.');
        trackResults = [];
        fclose(fid);
        %return;
    end
end