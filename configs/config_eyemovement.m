%% Loads parameters used for eye movement simulation.
% Selecting parameters using this config generates Figure 4(b) of paper.

%% Parameter of Algorithms.
% OLS bandit Parameters.
q = 1;
h = 1;
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
ns = 100;