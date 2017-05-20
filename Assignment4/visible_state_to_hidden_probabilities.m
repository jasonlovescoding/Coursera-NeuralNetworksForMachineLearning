function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> 
% by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> 
% by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and 
% returns the activation probabilities of the hidden units conditional on those states.
    %error('not yet implemented');
    number_of_hidden_units = size(rbm_w, 1);
    number_of_configurations = size(visible_state, 2);
    hidden_probability = zeros(number_of_hidden_units, number_of_configurations);
    
    for k = 1:number_of_configurations
        visible_state_k = visible_state(:, k);
        hidden_probability(:, k) = 1 ./ (1 + exp(-(rbm_w * visible_state_k)));
    end
end
