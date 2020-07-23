%% Loads parameters used for warfarin simulation.
% Selecting parameters using this config generates Figure 4(d) of paper.

%% Parameter of Algorithms.
% OLS bandit Parameters.
q = 1;
h = 5;
% OFUL parameters.
lambdaOFUL = 1;
deltaOFUL = 0.99;
% Prior-dependent Thompson Sampling parameters.
prior_mean = zeros(d, 1);
prior_cov = eye(d);
% Greedy-First parameters.
min_eig_threshold = 1e-20;
t0 = 4*k*d;
% Number of rounds of random sampling in the beginning (for Greedy and
% Greedy-First).
random_initialization = d;
%
%% Number of simulations.
% Takes a long time to run but generates Figure 4(d) of paper. Consider
% replacing it with ns = 10 for faster results (with wider confidence
% sets).
ns = 100;  
