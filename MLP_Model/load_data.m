function data_sets = load_data(data_file_name)
data_file_name = 'Z:\Long\MLP_Model\spoof_data_normalize_compress111-130.mat';
%data_file_name = 'D:\Long\MLP_Model\MNIST.mat';
% Description: Helper function to load data and structure it for use in
% MLP_Test
data_sets = load(data_file_name);
data_sets.training.bias_i = ones(data_sets.training_count, 1);
data_sets.validation.bias_i = ones(data_sets.validation_count, 1);
data_sets.test.bias_i = ones(data_sets.test_count, 1);
data_sets.training.bias_o = ones(data_sets.training_count, 1);
data_sets.validation.bias_o = ones(data_sets.validation_count, 1);
data_sets.test.bias_o = ones(data_sets.test_count, 1);
data_sets.training.count = data_sets.training_count;
data_sets.validation.count = data_sets.validation_count;
data_sets.test.count = data_sets.test_count;