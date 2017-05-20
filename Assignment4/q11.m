function partition = q11(small_test_rbm_w)
% In response to question 11, this function returns the normalizing constant 
% given a weight matrix between hidden units and visible units. 
% Thanks a lot to Chun Lam at Week13 forum!
% A way to understand this is to index the the partition's terms, with respect
% to the hidden units (write them in binary to make a binary number as index).
% Partition the terms into groups according to the binary number,
% and you will find that within each group, the partition term is the
% multiplication of all the terms in the form: 
% (1 + exp(rbm weight * hidden units' binary vector)), where
% rmb weight is a line vector corresponding to the state of the visible units
% hidden units' binary vector is just the index written as a vector of its digits
    % the binary combination matrix
    A = dec2bin(0:2^10-1) - '0';
    % sum over all the binary states of hidden units
    partition = log(sum(prod(1 + exp(small_test_rbm_w' * A'))));
end

