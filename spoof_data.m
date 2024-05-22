clear all;
sec=0;
training.input=zeros(0, 5250);
training.output=zeros(0, 2);
training.classes=zeros(0, 1);

test.input=zeros(0, 5250);
test.output=zeros(0, 2);
test.classes=zeros(0, 1);

validation.input=zeros(0, 5250);
validation.output=zeros(0, 2);
validation.classes=zeros(0, 1);

min_nonspoof1=0;
max_nonspoof1=8500;
min_nonspoof2=8501;
max_nonspoof2=9800;
min_nonspoof3=9801;
max_nonspoof3=11000;

min_spoof1=15001;
max_spoof1=18000;
min_spoof2=18001;
max_spoof2=19000;
min_spoof3=19001;
max_spoof3=20000;

for sec=[0:2977, 2979:11000, 11001:15000]
    %disp(sec);
    filename = sprintf('results_%d.mat', sec);
    load (filename);
    save_results_old=save_results;
    minValue=min(save_results_old(:));
    maxValue=max(save_results_old(:));
    save_results=(save_results_old-minValue)/(maxValue-minValue);
    if sec<=max_nonspoof3
        if sec<=max_nonspoof1 %sec<=30
            training.input = [training.input; save_results];
            for i=1:size(save_results, 1)
                training.output=[training.output; [1 0]];
                training.classes=[training.classes; ["Non-spoof"]];
            end
        elseif sec>=min_nonspoof2 && sec<=max_nonspoof2 %sec>30 && sec<=35
            test.input = [test.input; save_results];
            for i=1:size(save_results, 1)
                test.output=[test.output; [1 0]];
                test.classes=[test.classes; ["Non-spoof"]];
            end
        else %elseif sec>35 && sec<=40
            validation.input = [validation.input; save_results];
            for i=1:size(save_results, 1)
                validation.output=[validation.output; [1 0]];
                validation.classes=[validation.classes; ["Non-spoof"]];
            end
        end
    elseif sec>=min_spoof1
        if sec<=max_spoof1 %sec<=140
            training.input = [training.input; save_results];
            for i=1:size(save_results, 1)
                training.output=[training.output; [0 1]];
                training.classes=[training.classes; ["Spoof"]];
            end
        elseif sec>=min_spoof2 && sec<=max_spoof2 %sec>180 && sec<=190
            test.input = [test.input; save_results];
            for i=1:size(save_results, 1)
                test.output=[test.output; [0 1]];
                test.classes=[test.classes; ["Spoof"]];
            end
        else %elseif sec>145 && sec<=150
            validation.input = [validation.input; save_results];
            for i=1:size(save_results, 1)
                validation.output=[validation.output; [0 1]];
                validation.classes=[validation.classes; ["Spoof"]];
            end
        end
    end
end

input_count=5250;
output_count=2;
training_count=size(training.input, 1);
test_count=size(test.input, 1);
validation_count=size(validation.input, 1);

save('spoof_data_normalize_compress151-200.mat', 'input_count', 'output_count', 'training_count', 'test_count', 'validation_count', 'training', 'test', 'validation', '-v7.3');