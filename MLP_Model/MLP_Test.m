function MLP_Test
%% TRAIN MULTI-LAYER PERCEPTRON MODEL
% Description: This function trains a MLP object on data from a .mat file,
% as formatted as a structure with parameters as definedbelow. Example
% datasets (MNIST and Fashion_MNIST), already formatted, are provided in
% the 'Example Datasets' folder.
%
% Download example MNIST and Fashion_MNIST datasets at:
% https://drive.google.com/open?id=1n4T2mSxUq3kahM_qlyWNOj6YmSD_Mh8c
%
% Use the following format for dataset .mat file:
% dataset: structure containing input data. The struct must have the
% following elements:
%
%           data.input_count: a [1x1] scalar containing number of input 
%               features (m features)
%           data.output_count: a [1x1] scalar containging number of output
%               classes (k classes)
%
%           data.training_count: a [1x1] scalar containing number of data
%               points in training set (n_train)
%           data.test_count:  a [1x1] scalar containing number of data 
%               points in test set (n_test)
%           data.validation_count: a [1x1] scalar containing number of data
%               points in validation set (n_val)
%
%           data.training.input: a [n_train x m] array containing inputs of
%               the training set (m features, n_train data points)
%           data.training.output: a [n_train x k] array containing outputs 
%               of the training set (one-hot vectorized, n_train data 
%               points, k features)
%           data.training.classes: a [n_train x 1 ] array containing output
%               classes of the training set (n_train data points)
%
%           data.test.input: a [n_test x m] array containing inputs of the 
%               test set (m features, n_test data points)
%           data.test.output: a [n_test x k] array containing outputs of 
%               the test set (one-hot vectorized, n_test data points, 
%               k features)
%           data.test.classes: a [n_test x 1] array containing output 
%               classes of the test set (non-vectorized, n_test data points)
%
%           data.validation.input: a [n_val x m] array containing inputs of
%               the validation set (m features, n_val data points)
%           data.validation.output: a [n_val x k] array containing outputs
%               of the validation set (one-hot vectorized, n_val data 
%               points, k features)
%           data.validation.classes:  a [n_val x 1] array containing output
%               classes of the validation set (non-vectorized, n_val data
%               points)
clear all;clc;
% Load dataset and define features
dataset = 'MNIST'; %'Fashion_MNIST';     % Choose between MNIST and Fashion_MNIST
data = load_data(strcat(dataset,'.mat'));
% Parse the data from the dataset
% Training Data
X = data.training.input;
Y = data.training.output;
% Test Data
X_test = data.test.input;
Y_test = data.test.output;
% Validation Data
X_val = data.validation.input;
Y_val = data.validation.output;
% define input and output features
n_features = data.input_count;
n_output_features = data.output_count;
n_data = data.training_count;   % Number of samples in training set
n_test_data = data.test.count;  % Number of samples in test set
% Construct MLP architecture
% NOTE: Input layer and output layer must always have the same dimension as
% the number of input and output features, respectively. The included
% datasets already have bias terms, which is why the 'bias' option is set
% to 'false'.
network = MLPNet();
network.AddInputLayer(n_features,false);
%network.AddHiddenLayer(256,'leakyrelu',false);
%network.AddHiddenLayer(256,'leakyrelu',false);
network.AddHiddenLayer(102,'leakyrelu',false);
network.AddHiddenLayer(78,'leakyrelu',false);
network.AddHiddenLayer(70,'leakyrelu',false);
%network.AddOutputLayer(n_output_features,'softmax',false);
network.AddOutputLayer(n_output_features,'sigmoid',false);
network.NetParams('rate',0.0005,'momentum','adam','lossfun','crossentropy',...
    'regularization','L2');
network.trainable = true;
network.Summary();
% Training parameters
acc = 0;                        % pre-allocate training accuracy
n_batch = 128;                  % Size of the minibatch
max_epoch = 20;                 % Maximum number of epochs
max_batch_idx = floor(n_data/n_batch);          % Maximum batch index
max_num_batches = max_batch_idx.*max_epoch;     % Maximum number of batches
% Pre-allocate for epoch and error vectors (for max iteration)
epoch = zeros(1,max_num_batches-1);
d_loss = epoch;
ce_test = zeros(max_epoch,1);
ce_train = zeros(max_epoch,1);
ce_val = zeros(max_epoch,1);
% Initialize iterator and timer
batch_idx = 1;      % Index to keep track of minibatches
epoch_idx = 1;      % Index to keep track of epochs
target_accuracy = 98; % Desired classification accuracy
while ((epoch(batch_idx)<max_epoch)&&(acc<target_accuracy))
    % Compute current epoch
    epoch(batch_idx+1) = batch_idx*n_batch/n_data;
    % Randomly sample data to create a minibatch
    rand_ind = randsample(n_data,n_batch);
    % Index into input and output data for minibatch
    X_batch = X(rand_ind,:);    % Sample Input layer
    Y_batch = Y(rand_ind,:);    % Sample Output layer
    
    % Train model
    d_loss(batch_idx+1) = network.training(X_batch,Y_batch)./n_batch;
    
    % Only compute error/classification metrics after each epoch
    if ~(mod(batch_idx,max_batch_idx))
        % Compute error metrics for training, test, and validation set
        [~,ce_train(epoch_idx),~]=network.NetworkError(X,Y,'classification');
        [~,ce_val(epoch_idx),~]=network.NetworkError(X_val,Y_val,'classification');
        tic;
        [~,ce_test(epoch_idx),~]=network.NetworkError(X_test,Y_test,'classification');
        eval_time = toc;
        fprintf('\n-----------End of Epoch %i------------\n', epoch_idx);
        fprintf('Loss function: %f \n',d_loss(batch_idx+1));
        fprintf('Test Set Accuracy: %f Training Set Accuracy: %f \n',1-ce_test(epoch_idx),1-ce_train(epoch_idx));
        fprintf('Test Set Evaluation Time: %f s\n\n',eval_time);
        acc = (1-ce_test(epoch_idx));
        epoch_idx = epoch_idx+1;    % Update epoch index
    end
    % Update batch index
    batch_idx = batch_idx+1;
end
% Remove trailing zeros if training met target accuracy before maximum
% number of epochs
ce_test = ce_test(1:(epoch_idx-1));
ce_train = ce_train(1:(epoch_idx-1));
ce_val = ce_val(1:(epoch_idx-1));
% Plot classification results
figure(1)
plot(ce_test);hold on;
plot(ce_train);hold on;
plot(ce_val);hold off;
grid on;
xlabel('Epoch');
ylabel('Classification Error');
legend('Test Set','Training Set','Validation Set');

end