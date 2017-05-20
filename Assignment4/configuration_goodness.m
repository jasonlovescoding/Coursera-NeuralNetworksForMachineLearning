function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> 
% by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> 
% by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) 
% of the described configurations.
    %error('not yet implemented');
    number_of_configurations = size(visible_state, 2);
    
    G = 0;
    for k = 1:number_of_configurations
        visible_state_k = visible_state(:, k);
        hidden_state_k = hidden_state(:, k);
        G = G + hidden_state_k' * rbm_w * visible_state_k;
    end
    G = G / number_of_configurations;
end
