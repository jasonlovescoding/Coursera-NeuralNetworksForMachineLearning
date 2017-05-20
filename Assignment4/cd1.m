% this is the version for question 8 & 9
% no longer do that sampling at the hidden state that results from 
% the "reconstruction" visible state. Instead of a sampled state, 
% we'll simply use the conditional probabilities.
function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size 
% <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. 
% It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    
    % Q9: add this line of code as the new first line of the cd1 function
    visible_data = sample_bernoulli(visible_data);
    
    % given the visible data, calculate the hidden probabilities
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  
    % sample the hidden states given its probabilities
    hidden_sample = sample_bernoulli(hidden_probabilities);
    
    % calculate the goodness's gradient with visible data and hidden sample
    data_goodness_gradient = configuration_goodness_gradient(visible_data, hidden_sample);
    
    % reconstruct the visible state probabilities with hidden sample
    visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);
    
    % sample the visible states given its probabilities
    visible_sample = sample_bernoulli(visible_probabilities);
    
    % calculate hidden probabilities again with reconstructed visible sample
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_sample);
    
    % calculate the goodness's gradient with reconstructed visible data and hidden sample 
    reconstruction_goodness_gradient = configuration_goodness_gradient(visible_sample, hidden_probabilities);
    
    ret = data_goodness_gradient - reconstruction_goodness_gradient;
end

% this is the version for question 7 
% (do the sampling every time once getting a probability)
function ret = q7_cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size 
% <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. 
% It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    
    % given the visible data, calculate the hidden probabilities
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  
    % sample the hidden states given its probabilities
    hidden_sample = sample_bernoulli(hidden_probabilities);
    
    % calculate the goodness's gradient with visible data and hidden sample
    data_goodness_gradient = configuration_goodness_gradient(visible_data, hidden_sample);
    
    % reconstruct the visible state probabilities with hidden sample
    visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);
    
    % sample the visible states given its probabilities
    visible_sample = sample_bernoulli(visible_probabilities);
    
    % calculate hidden probabilities again with reconstructed visible sample
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_sample);
    
    % sample the hidden states given its probabilities 
    hidden_sample = sample_bernoulli(hidden_probabilities);
    
    % calculate the goodness's gradient with reconstructed visible data and hidden sample 
    reconstruction_goodness_gradient = configuration_goodness_gradient(visible_sample, hidden_sample);
    
    ret = data_goodness_gradient - reconstruction_goodness_gradient;
end
