classdef MLPNet < handle
% Description: This class defines a framework for sequential construction
% and training of a multi-layer perceptron network (or deep neural network)
%
% AVAILABLE PARAMETERS:
%   activation functions: tanh, relu, leakyrelu, sigmoid, softmax
%   loss functions: rmse, cross-entropy, binary cross-entropy
%   regularization: L1, L2, dropout
%   momentum: none, regular, ADAM, Nesterov
%   annealing: none, step, exponential, 1/t
%
% EXAMPLE USAGE
% % Construct DNN architecture
% network = MLPNet();
% network.AddInputLayer(n_features,false);
% network.AddHiddenLayer(128,'tanh',false);
% network.AddHiddenLayer(128,'tanh',false);
% % network.AddHiddenLayer(128,'tanh',false);
% network.AddOutputLayer(n_output_features,'tanh',false);
% network.NetParams('rate',0.0005,'momentum','adam','lossfun','rmse',...
%     'regularization','none');
% network.trainable = true;
% network.Summary();
%
% network.training(X_train,Y_train)
%
% % Test the network and compute error metrics
% Y_sample_pred = network.test(X_sample);
%
% % Get error metrics
% network.NetworkError(X_test,Y_test,'classification');
 
    % Define class parameters
    properties (Access = public)
        Layers = {};        % cell containing number of nodes in each layer
        Weights = {};       % cell containing weights between layers
        Nets = {};          % cell containing network outputs               
        H = {};             % cell containing output activations
        Act_Fun = {};       % cell containing activation function
        lossfun = 'rmse';       % Default Loss Function
        trainable = true;    % boolean for trainability
        NL=0;                % Number of layers in network
        regularization = 'none';    % Defaault Regularization Methods
        momentum = 'adam';          % Default Type of momentum
        annealing = 'none';         % Default Type of annealing
        rate = 0.0005;              % Default Learning Rate
        net_bias = [];      % Bias node applied to each network
        dropout = 1;        % Default dropout rate
        mask = {};          % Dropout mask 
    end
    
    properties (Constant)
        mu = 0.5;           % For regular/Nesterov momentum
        beta1 = 0.9;        % For ADAM
        beta2 = 0.99;       % For ADAM
        eps = 1E-8;         % For ADAM
        alpha = 0.01;       % For LeakyRelu
        lambda = 1E-6;      % Regularization Parameter
    end
    
    properties (Access = private)
        m = {};             % ADAM parameters
        v = {};             % ADAM parameters
        v_prev = {};        % ADAM parameters
        init = [];          % ADAM parameters
    end
    
    % Define class methods
    methods
        function obj = Summary(obj)
            % Description: This function prints the network architecture
            % and algorithm parameters.
            % INPUTS:
            %   obj: class handle
            %
            % OUTPUTS: 
            %   obj: return class handle
            
            fprintf('Network Type: MLP\n');
            fprintf('Loss: %s\n',obj.lossfun);
            fprintf('Momentum: %s\n',obj.momentum);
            fprintf('Regularization: %s\n',obj.regularization);
            fprintf('Learning Rate: %f\n',obj.rate);
            fprintf('Dropout Rate (0=none): p=%f\n',obj.dropout);
            fprintf('Trainable: %i \n',obj.trainable);
            fprintf('-------Network Architecture--------\n');
            fprintf('Input Layer: \t %i Neurons\n',obj.Layers{1});
            for i = 2:obj.NL
                fprintf('Layer %i: \t %i Neurons \t %s Activation\n',...
                    i,obj.Layers{i},obj.Act_Fun{i-1});
            end
            fprintf('\n\n');
        end
        function obj = NetParams(obj, varargin)
            % Description: Load network algorithm parameters
            % varargin: value-pair options for running algorithm
            %           {'rate', [1x1 scalar]}: Learning rate
            %           {'annealing', [string]}: rate annealing (possible options:
            %               {'none','step','exponential','1/t'}
            %           {'momentum', [string]}: momentum (possible options:
            %               {'none','regular','nesterov','adam'}
            %           {'regularization', [string]}: regularization (possible options:
            %               {'none','dropout','L2','L1'}
            %           {'loss', [string]}: type of loss function used
            %               possible options:
            %               {'rmse','crossentropy','binarycrossentropy'}
            %           {'dropout', [string]}: dropout parameter. If not
            %               specified, then no dropout occurs (p = 1)
            
            % Throw error if trying to set network parameters before
            % constructing the network
            if isempty(obj.Layers)
                error('Must construct network architecture before setting parameters');
            end
            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj');
            addOptional(p, 'momentum', 'none', ...
                @(s) any(validatestring(s, {'none','regular','nesterov','adam'})));
            addOptional(p, 'regularization', 'none', ...
                @(s) any(validatestring(s, {'none','L1','L2'})));
            addOptional(p, 'dropout', obj.dropout, ...
                @(x) isnumeric(x) && isscalar(x) && (x>0) && (x<=1));
            addOptional(p, 'annealing', 'none', ...
                @(s) any(validatestring(s, {'none','step','exponential','1/t'})));
            addOptional(p, 'rate', obj.rate, ...
                @(x) isnumeric(x) && isscalar(x) && (x>0));
            addOptional(p, 'lossfun', obj.lossfun,...
                @(s) any(validatestring(s,{'rmse','crossentropy','binarycrossentropy'})));
            parse(p,dataset,varargin{:});
            % Assign new inputs to MLPNet class parameters
            obj.momentum = p.Results.momentum;
            obj.regularization = p.Results.regularization;
            obj.annealing = p.Results.annealing;
            obj.rate = p.Results.rate;
            obj.lossfun = p.Results.lossfun;
            obj.dropout = p.Results.dropout;
            
            if (strcmp(obj.lossfun,'crossentropy')||strcmp(obj.lossfun,'binarycrossentropy')) &&...
                (strcmp(obj.Act_Fun{end},'tanh')||strcmp(obj.Act_Fun{end},'relu')||...
                strcmp(obj.Act_Fun{end},'leakyrelu'))
                disp('ERROR!')
                error_str = strcat('Cannot use ',obj.Act_Fun{end},...
                    ' with ',obj.lossfun,' loss function. Use sigmoid or softmax instead');
                error(error_str);
            end
            
        end
        
        function obj = AddInputLayer(obj,n_features,bias)
            % Description: add an input layer to the network
            % INPUTS:
            %   obj: class handle
            %   n_features: number of input features [1x1 scalar]
            %   activation_function: type of activation function applied to
            %       input layer [string]
            %   bias: Include bias node [boolean]
            %
            % OUTPUTS: 
            %   obj: return class handle
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj');
            addRequired(p, 'n_features',...
                @(x) isnumeric(x) && isscalar(x) && (x>0));
            addRequired(p, 'bias',...
                @(x) islogical(x));
            parse(p, obj, n_features, bias);
      
            % Update network features
            obj.NL = obj.NL+1;
            obj.H{1} = ones(n_features+boolean(bias),1);
            obj.Layers{1} = n_features+boolean(bias);
            obj.net_bias{1} = boolean(bias);
        end
        
        function obj = AddHiddenLayer(obj,N_n,activation_function,bias)
            % Description: add a hidden layer to the network
            % INPUTS:
            %   obj: class handle
            %   N_n: number of hidden nodes in layer [1x1 scalar]
            %   activation_function: type of activation function applied to
            %       input layer [string]
            %   bias: Include bias node [boolean]
            %
            % OUTPUTS: 
            %   obj: return class handle
            
            % Throw error if trying to define hidden layer before input
            % layer
            if isempty(obj.Layers)
                error('Must have an input layer before adding hidden layers');
            end
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj');
            addRequired(p, 'N_n',...
                @(x) isnumeric(x) && isscalar(x) && (x>0));
            addRequired(p, 'activation_function', ...
                @(s) any(validatestring(s,{'tanh','relu','leakyrelu',...
                'sigmoid','softmax'})));
            addRequired(p, 'bias',...
                @(x) islogical(x));
            parse(p, obj, N_n, activation_function, bias);
            
            % Update network features
            obj.NL = obj.NL+1;
            obj.Layers{obj.NL} = N_n+boolean(bias);
            obj.H{obj.NL} = ones(N_n+boolean(bias),1);
            obj.net_bias{obj.NL} = boolean(bias);
            obj.Act_Fun{obj.NL-1} = activation_function;
            obj.Nets{obj.NL-1} = ones(obj.Layers{obj.NL-1}+boolean(bias),1);
            obj.Weights{obj.NL-1} = .5 - rand(obj.Layers{obj.NL},...
                obj.Layers{obj.NL-1});
        end
        
        function obj = AddOutputLayer(obj,n_output_features,activation_function,bias)
            % Description: add an output layer to the network
            % INPUTS:
            %   obj: class handle
            %   n_output_features: number of output features [1x1 scalar]
            %   activation_function: type of activation function applied to
            %       input layer [string]
            %   bias: Include bias node [boolean]
            %
            % OUTPUTS: 
            %   obj: return class handle
            
            % Throw error if trying to define hidden layer before input
            % layer
            if isempty(obj.Layers)
                error('Must have an input layer before adding output layer');
            end
            
            % Parse inputs
            p = inputParser;
            addRequired(p, 'obj');
            addRequired(p, 'n_output_features',...
                @(x) isnumeric(x) && isscalar(x) && (x>0));
            addRequired(p, 'activation_function', ...
                @(s) any(validatestring(s,{'tanh','relu','leakyrelu',...
                'sigmoid','softmax'})));
            addRequired(p, 'bias', ...
                @(x) islogical(x));
            parse(p, obj, n_output_features, activation_function, bias);
            
            % Throw error if output activation is not compatible with
            % desired loss function
            if (strcmp(obj.lossfun,'crossentropy')||strcmp(obj.lossfun,'binarycrossentropy')) &&...
                (strcmp(activation_function,'tanh')||strcmp(activation_function,'relu')||...
                strcmp(activation_function,'leakyrelu'))
            
                error_str = strcat('Cannot use ',activation_function,...
                    ' with ',obj.lossfun,' loss function. Use sigmoid or softmax instead');
                error(error_str);
            end
            
            % Update network features
            obj.NL = obj.NL+1;
            obj.Layers{obj.NL} = n_output_features+boolean(bias);
            obj.H{obj.NL} = ones(n_output_features+boolean(bias),1);
            obj.net_bias{obj.NL} = boolean(bias);
            obj.Act_Fun{obj.NL-1} = activation_function;
            obj.Nets{obj.NL-1} = ones(obj.Layers{obj.NL-1},1);
            obj.Weights{obj.NL-1} = .5 - rand(obj.Layers{obj.NL},...
                obj.Layers{obj.NL-1});
        end
        
        function obj = network_output(obj,layer)
            % Description: compute network output for specified layer 
            % INPUTS:
            %   obj: class handle
            %   layer: current layer [1x1 scalar]
            %
            % OUTPUTS: 
            %   obj: return class handle
            
            % unregularized network output
            obj.Nets{layer} = (obj.Weights{layer}*obj.H{layer}')';       
        end
        
        function Y_hat = feedforward(obj, X)
            % Description: feedforward through the network with dropout
            % INPUTS:
            %   obj: class handle
            %   X: input data [struct]
            %
            % OUTPUTS: 
            %   Y_hat: predicted output
            
            % First activation layer is input layer
            obj.H{1} = X;
            
            %--------------------------------------------------------------
            % Feedforward to get net outputs and activation for all layers
            %--------------------------------------------------------------
            for i=1:obj.NL-1
                % Network output from previous layer
                obj.network_output(i);
                obj.activate(i);
            end
            
            % Output estimate
            Y_hat = obj.H{end};    
        end
        
        function Y_hat = test_from_layer(obj,layer,X_L)
            % Description: predict output from internal layer (for use in
            % Autoencoder/VAE
            %
            % INPUTS:
            %   obj: class handle
            %   layer: layer to feedforward from [1x1 scalar]
            %   X_L: input [must have same dimension as nodes in layer]
            %
            % OUTPUTS: 
            %   Y_hat: predicted output
            
            obj.Nets{layer} = X_L;
            obj.activate(layer);
            %--------------------------------------------------------------
            % Feedforward to get net outputs and activation for all layers
            %--------------------------------------------------------------
            for i=layer+1:obj.NL-1
                % Network output from previous layer
                obj.Nets{i} = (obj.Weights{i}*obj.H{i}')';
                obj.activate(i);
            end
            % Output estimate
            Y_hat = obj.H{end}; 
        end
        function Y_hat = test(obj, X)
            % Description: predict output (no dropout)
            % INPUTS:
            %   obj: class handle
            %   X: input data [struct]
            %
            % OUTPUTS: 
            %   Y_hat: predicted output
            obj.H{1} = X;
            
            %--------------------------------------------------------------
            % Feedforward to get net outputs and activation for all layers
            %--------------------------------------------------------------
            for i=1:obj.NL-1
                % Network output from previous layer
                obj.Nets{i} = (obj.Weights{i}*obj.H{i}')';
                obj.activate(i);
            end
            
            % Output estimate
            Y_hat = obj.H{end}; 
        end
            
        
        function [sigma, delta] = backprop(obj, target, Y_hat)
            % Description: backpropagate from output layer
            % INPUTS:
            %   obj: class handle
            %   target: target outputs [n_data x n_output_features]
            %   Y_hat: predicted output [n_data x n_output_features]
            %
            % OUTPUTS: 
            %   sigma: derivative of first layer activation
            %   delta: delta for first layer
            % Compute error vector
            % activation gradients and deltas for each layer
            sigma = obj.dactivate(obj.NL-1);
            switch obj.lossfun
                case 'rmse'
                    % derivative of cost function
                    error_vector = (target - Y_hat);
                    delta = error_vector.*sigma;
                case 'crossentropy'
                    %derivative of cost function
                    error_vector = (target - Y_hat);
                    delta = error_vector;
                case 'binarycrossentropy'
                    %derivative of cost function
                    error_vector = (target - Y_hat);
                    delta = error_vector;
            end
            % %------------------------------------------------------------
            % % Backpropagate to adjust weights in hidden layer
            % %------------------------------------------------------------
            % Backpropagate through dropout mask
            for i=obj.NL-1:-1:1
                if obj.trainable
                    dw_mean = delta'*obj.H{i};  % Sum of all delta activations
                    obj.UpdateWeights(dw_mean,i); 
                end
                if i > 1
                    % Update sum of delta activations for next layer
                    % Only pass through dropout if training
                    if obj.trainable
                        sigma = (obj.mask{i-1}).*obj.dactivate(i-1);
                    else
                        sigma = obj.dactivate(i-1);
                    end
                    delta = (delta*obj.Weights{i}).*sigma;
                end
            end
        end
        
        function loss = training(obj, X, target)
            % Description: Train the network
            % INPUTS:
            %   obj: class handle
            %   X: training data [n_batch x n_features]
            %   target: target outputs [n_batch x n_output_features]
            %
            % OUTPUTS: 
            %   loss: return loss function
            
            % Feedforward through dropout mask
            Y_hat = obj.feedforward(X);
            
            switch obj.lossfun
                case 'rmse'
                    loss = 0.5*(sum(sum(target-Y_hat,1))).^2;
                case 'crossentropy'
                    loss = -sum(sum(target.*log(Y_hat),1));
                case 'binarycrossentropy'
                    loss = -sum(sum(target.*log(Y_hat)+(1-target).*log(1-Y_hat),1));
            end
            
            % backpropagate through dropout mask
            obj.backprop(target, Y_hat);
        end
        
        function obj = UpdateWeights(obj,delta_w,layer)
            % Description: This function updates the weights during the
            % backward pass
            %
            % INPUTS:
            %   obj: class handle
            %   delta_w: weight update [n_{layer-1} x n{layer}]
            %   layer: current layer [1x1 scalar]
            %
            % OUTPUTS:
            %   obj: return class handle
            
            % If first initialization, allocate space
            if ~strcmp(obj.momentum,'none')&&isempty(obj.init)
                obj.v = cell(obj.NL);
                obj.v_prev = cell(obj.NL);
                obj.m = cell(obj.NL);
                obj.init = 1;
            end
            
            % If first time accessing current layer, initialize place in cell
            if ~strcmp(obj.momentum,'none')&&isempty(obj.v{layer})
                obj.v{layer} = zeros(size(delta_w));
                obj.v_prev{layer} = zeros(size(delta_w));
                obj.m{layer} = zeros(size(delta_w));
            end
            
            % Update weights depending on desired momentum method
            switch obj.momentum
                case 'regular'
                    obj.v{layer} = -obj.mu.*obj.v{layer}+obj.rate.*delta_w;
                    obj.Weights{layer} = obj.Weights{layer} +  obj.v{layer};
                case 'nesterov'
                    obj.v_prev{layer} = obj.v{layer};
                    obj.v{layer} = -obj.mu*obj.v{layer} + obj.rate.*delta_w;
                    obj.Weights{layer} = obj.Weights{layer}...
                        + obj.mu*obj.v_prev{layer} + (1+obj.mu).*obj.v{layer};
                case 'adam'
                    obj.m{layer} = obj.beta1.*obj.m{layer} + (1-obj.beta1).*delta_w;
                    obj.v{layer} = obj.beta2.*obj.v{layer} + (1-obj.beta2).*(delta_w.^2);
                    obj.Weights{layer} = obj.Weights{layer}...
                        + obj.rate.*obj.m{layer}./(sqrt(obj.v{layer})+obj.eps);
                case 'none'
                    prev_w = obj.Weights{layer};
                    obj.Weights{layer} = prev_w + obj.rate.*delta_w;
            end
            
            
            % Modify gradient update depending on regularization method
            switch obj.regularization
                case 'L2'
                    w_in = obj.Weights{layer};
                    obj.Weights{layer}= w_in+obj.lambda.*w_in;
                case 'L1'
                     w_in = obj.Weights{layer};
                     obj.Weights{layer}= w_in+obj.lambda.*sign(w_in);
                otherwise
                    obj.Weights{layer} = obj.Weights{layer};
            end
        end
        
        function act_out = activate(obj, layer)
            % Description: This function applies the activation function element-wise
            % to X
            %
            % INPUTS:
            %   obj: class handle
            %   layer: current layer to be activated [1x1 scalar]
            % OUTPUTS:
            %   obj: return class handle
                     
            % Pass network output through appropriate activation function
            switch obj.Act_Fun{layer}
                case 'tanh'
                    act_out = tanh(obj.Nets{layer});
                case 'sigmoid'
                    act_out = 1./(1+exp(-obj.Nets{layer}));
                case 'relu'
                    act_out = max(zeros(size(obj.Nets{layer})),...
                        obj.Nets{layer});
                case 'leakyrelu'
                    act_out = obj.alpha.*(obj.Nets{layer}<0)...
                        .*obj.Nets{layer}+(obj.Nets{layer}>0)...
                        .*obj.Nets{layer};
                case 'softmax'
                    act_out = exp(obj.Nets{layer})...
                        ./repmat(sum(exp(obj.Nets{layer}),2),1,size(obj.Nets{layer},2));
            end
            
            % Apply dropout to hidden layers
            if (layer<obj.NL-1)
                if obj.dropout<1
                    % Create dropout mask
                    obj.mask{layer} = (rand(size(act_out))<(obj.dropout))...
                        ./(obj.dropout);
                else
                    % Normalize mask if no dropout desired
                    obj.mask{layer} = ones(size(act_out));
                end
                obj.H{layer+1} = obj.mask{layer}.*act_out;
            else
                obj.H{layer+1} = act_out;
            end
        end
        
        function sigma = dactivate(obj,layer)
            % Description: This function applies the derivative of the activation
            %function element-wise to X
            %
            % INPUTS:
            % layer: current layer [1x1 scalar]
            % obj: class handle
            %
            % OUTPUTS:
            % sigma: the derivative of the activated nodal outputs [nxm matrix]
            
            % Update weights depending on desired activation function
            switch obj.Act_Fun{layer}
                case 'tanh'
                    sigma = (1-tanh(obj.Nets{layer}).^2);
                case 'sigmoid'
                    P = 1./(1+exp(-obj.Nets{layer}));
                    sigma = P.*(1-P);
                case 'relu'
                    sigma = obj.Nets{layer}>0;
                case 'leakyrelu'
                    sigma = obj.alpha.*(obj.Nets{layer}<0)+(obj.Nets{layer}>0);
                case 'linear'
                    sigma = ones(size(obj.Nets{layer}));
                case 'softmax'
                    P = obj.activate(layer);
                    sigma = P.*(1-P);
                    
            end
        end
        
        % Function for computing error/accuracy metrics
        function [ERROR,CE,ACC] = NetworkError(obj,X,Y,type)
            % Description: This function computes the network error (and
            % classification error/accuracy for classification problems)
            %
            % INPUTS:
            % X: input [n_data x n_input_features]
            % Y: output [n_data x n_output_features]
            % type: string {classification,regression}
            %
            % OUTPUTS:
            % ERROR: network error [1x1 scalar]
            % CE: classification error [1x1 scalar]
            % ACC: classification accuracy [1x1 scalar]
 
            n_data = size(X,1);
            
            % add bias node to input if input layer has bias
            if obj.net_bias{1}
                X = horzcat(X, ones(n_data,1));
            end
            
            Y_hat = obj.test(X);
            % compute error
            [~,output_count] = size(Y);
            
            switch obj.lossfun
                case 'rmse'
                    ERROR = sum(sum((Y_hat-Y) .^2))/(n_data * output_count);
                case 'crossentropy'
                    ERROR = -sum(sum(Y.*log(Y_hat),1))/(n_data * output_count);
                case 'binarycrossentropy'
                    ERROR = -sum(sum(Y.*log(Y_hat)+(1-Y).*log(1-Y_hat),1)/(n_data * output_count));
            end
            
            % classification error
            % if problem is a classifcation problem, also output
            % classification error and accuracy. Otherwise, output zero
            if strcmp(type,'classification')
                % Convert outputs (one-hot vectors) to classes (scalar)
                [~,classes] = max(Y_hat, [], 2);
                [~,target_classes] = max(Y, [], 2);
                CE = sum(classes ~= target_classes)/n_data;
                ACC = 100*sum(classes == target_classes)/n_data;
            else
                CE = 0;
                ACC = 0;
            end
        end
    end
end
