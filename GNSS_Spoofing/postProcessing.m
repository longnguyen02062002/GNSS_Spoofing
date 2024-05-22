%% Initialization
disp ('Starting processing...');

[fid, message] = fopen (settings.filename, 'rb');

% If success, then process the data
if (fid > 0)
    % Move the starting point of processing. Can be used to start the signal processing at any point
    % in the data record (e.g. good for long records or for signal processing in blocks).
    fseek (fid, settings.skipNumberOfBytes, 'bof');

%% Acquisition
    % Do acquisition if it is not disabled in settings or if the variable acqResults does not exist.
    if ((settings.skipAcquisition == 0) || ~exist ('acqResults', 'var'))
        % Find number of samples per spreading code
        samplesPerCode = round(settings.samplingFreq / (settings.codeFreqBasis / settings.codeLength));

        % Read data for acquisition. 11ms of signal are needed for the fine frequency estimation
        data = fread (fid, 11*samplesPerCode, settings.dataType);

        % Do the acquisition
        disp ('   Acquiring satellites...');
        acqResults = acquisition(data, settings);
        plotAcquisition(acqResults);
    end

%% Initialize channels and prepare for the run
    % Start further processing only if a GNSS signal was acquired (the
    % field FREQUENCY will be set to 0 for all not acquired signals)
    if (any(acqResults.carrFreq))
        channel = preRun (acqResults, settings);
        showChannelStatus(channel, settings);
    else
        % No satellites to track, exit
        disp('No GNSS signals detected, signal processing finished.');
        trackResults = [];
        return;
    end

else
    % Error while opening the data file.
    error ('Unable to read file %s: %s.', settings.fileName, message);
end % if (fid > 0)
