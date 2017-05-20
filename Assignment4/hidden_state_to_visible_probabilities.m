function visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <hidden_state> is a binary matrix of size <number of hidden units> 
% by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of visible units> 
% by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the hidden units, and 
% returns the activation probabilities of the visible units, conditional on those states.
    %error('not yet implemented');
    number_of_visible_units = size(rbm_w, 2);
    number_of_configurations = size(hidden_state, 2);
    visible_probability = zeros(number_of_visible_units, number_of_configurations);
    
    for k = 1:number_of_configurations
        hidden_state_k = hidden_state(:, k);
        visible_probability(:, k) = 1 ./ (1 + exp(-(hidden_state_k' * rbm_w)))';
    end
end
