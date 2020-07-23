%% runoracle.m

% Trains the best linear model in hindsight and computes its regret.

% This function trains the oracle, i.e., best linear model having all 
% observations available and calculates the regret of this oracle
% algorithm.

%% Inputs:
%   k: Number of arms.
%   T: Time horizon.
%   d: Dimension of covariates.
%   verbose: Whether to print outputs or not.
%   X: A T*d matrix containing all contexts.
%   rewards: A T*k matrix, containing the rewards of all actions at all
%   time periods.
%
%% Outputs:
%
%   regret: Cumulative regret as a running sum over regret terms.
%   fractions: Fractions of pulls of different arms.
%

function [regret, fractions] = runoracle(k, T, d, verbose, X, rewards)

regret = zeros(1,T);
dec = zeros(T,k);

beta = zeros(d,k);

for j=1:k
    beta(:,j) = X\rewards(:,j);
end

for i=1:T
    x = X(i,:);
    
    [~, ind] = max(x*beta);
    ourreward = rewards(i, ind);
    bestreward = max(rewards(i, :));
    
    if (i==1)
        regret(i) = bestreward - ourreward;
    else
        regret(i) = regret(i-1) + bestreward - ourreward;
    end

end
fractions = mean(dec);  %fraction of times each arm is pulled

if(verbose == 1)
    fprintf('Oracle: Fraction of pulls = %f. \n', fractions);
    fprintf('Oracle: Total regret occured = %f. \n', regret(end));
end

end

