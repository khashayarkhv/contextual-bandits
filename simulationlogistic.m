%% Simulations with logistic rewards. 

% Setting low_dimensional = 1 creates Figure 3(a) and setting
% low_dimensional = 0 creates Figure 3(b).

%% Clear the workspace and close all figures.
close all;
clear;
random_seed = 10;
rng(random_seed);
tic
addpath('scripts');
%
%% GETTING FIGURE (a) [low_dimensional = 1] OR (b) [low_dimensional = 0].
low_dimensional = 1;  
%
%% Problem parameters.
ns = 10;   % Number of simulation runs.
k = 2;  % Number of Arms.
T = 2000;  % Length of each simulation.

dim_disc = 0;  % Dimension of binary distribution.

if(low_dimensional==1)
    dim_cont = 3;  % Dimension of continuous distribution.
    xmax = 1;
else
    dim_cont = 10;  % Dimension of continuous distribution.
    xmax = 5;
end

verbose = 1;  % Print results in each iteration or not.

save_figure = 0;  % Whether to save figures.
save_data = 0;    % Whether to save data. Note that the data is large. 

d = dim_cont + dim_disc;   % Total dimension.

%% Parameter of algorithms.
% GLM-UCB parameters.
deltaUCB = 0.01;
% Number of rounds of random sampling in the beginning (for Greedy and
% Greedy-First).
random_initialization = d;
%
%% Context and noise specifications.
cont_uplim = xmax;  % Gaussian upper truncation limit.
cont_lowlim = -xmax;  % Gaussian lower truncation limit.
disc_lowlim = -xmax;  % Generates Rademacher random variables. Unused here.
disc_uplim = xmax;  % Generates Rademacher random variables. Unused here.
disc_lowlim_prob = 0.5;  % Generates Rademacher random variables. Unused here.

sigma_x = 0.5 * eye(dim_cont);   % Covariance matrix of gaussian contexts.


%% Matrices for saving regrets.
reg_gb = zeros(ns, T);
reg_glmucb = zeros(ns, T);

%% Matrices for saving fraction of pulls.
frac_gb = zeros(ns, k);
frac_glmucb = zeros(ns, k);

%% Main code, run all algorithms.

for s=1:ns
    % Generate arm parameters.
    b = randn(k, d);
    prior_mean = zeros(d, 1);
    prior_cov = eye(d, d);
    
    % Generate continuous covariates from truncated Gaussian distribution.
    Xcont = max(cont_lowlim, min(cont_uplim, ...
        mvnrnd(zeros(dim_cont ,1), sigma_x, T)));
    
    % Generate discrete covariates from Rademacher distribution. Unused
    % here as dim_disc = 0 in our simulations.
    Xdisc = (disc_uplim - disc_lowlim) * ...
        (rand(T, dim_disc) < disc_lowlim_prob) + disc_lowlim;
    
    X = [Xcont, Xdisc];
    for i=1:T
        if(norm(X(i,:)) > xmax)
            X(i,:) = xmax * X(i,:)/norm(X(i,:));
        end
    end
    
    % Generate logistic rewards.
    reward_vector = zeros(T, k);
    
    for i=1:T
        for j=1:k
            reward_vector(i, j) = rand < exp(b(j, :) * X(i, :)') / ...
                (1 + exp(b(j, :) * X(i, :)'));
        end
    end


    fprintf('Round %d started. \n', s);
    
    [tmp_reg_ucb, tmp_frac_ucb] = runGLMUCB(k, T, d, b, xmax, ...
        deltaUCB, X, reward_vector, verbose);
    
    [tmp_reg_gb, tmp_frac_gb] = rungreedybanditlogistic(k, T, d, b, ...
        random_initialization, X, reward_vector, verbose);
    
    reg_glmucb(s,:) = tmp_reg_ucb;
    reg_gb(s,:)=tmp_reg_gb;
    
    fprintf('Total Regret after %d simulations \n', s)
    fprintf('GLM-GB=%f \n', mean(reg_gb(1:s, end)))
    fprintf('GLM-UCB=%f \n', mean(reg_glmucb(1:s, end)))
    
    frac_gb(s,:) = tmp_frac_gb;
    frac_glmucb(s,:) = tmp_frac_ucb;
    toc
end

%% Save the results and create plots.

% Generate file name.
if(low_dimensional==1)
    namedata = strcat('Synth_logistic_lowdim_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_d_', num2str(d),'_xmax_', num2str(xmax));
else
    namedata = strcat('Synth_logistic_highdim_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_d_', num2str(d),'_xmax_', num2str(xmax));
end
        

if(save_data==1)
    if ~exist('results/data/', 'dir')
       mkdir('results/data/')
    end
    eval(['save results/data/', namedata]);
end
% 
close all;
nP = 50;
st = 20;
step = floor(T/nP);
subint = st: step: T;
if (T<nP)
    subint = 1:T;
end

fontName ='Times New Roman';
fontSize = 16;
fontWeight ='bold';
addpath('shadedErrorBar');

h1=figure;
hold on;

DarkPastelGreen = [112 173 71]/255;
DarkBlue = [68 114 196]/255;

C=shadedErrorBar(subint, reg_glmucb(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'-.', 'Linewidth', 2, 'Color', DarkBlue}, 1);
E=shadedErrorBar(subint, reg_gb(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'Linewidth', 2, 'Color', DarkPastelGreen}, 1);


l = legend([C.mainLine, E.mainLine], 'GLM-UCB', 'GLM-GB', ...
    'Location', 'northeast');

set(l, 'interpreter', 'latex')

grid on;
set(gca, 'FontSize', fontSize, 'FontName', fontName)
ylabel('$\textrm{Regret}(t)$', 'Interpreter', 'latex');
xlabel('t');

if (save_figure==1)
    if ~exist('results/figures/', 'dir')
       mkdir('results/figures/')
    end
    set(h1, 'PaperUnits', 'inches', 'PaperPosition', [0 0 10 10]);
    saveas(h1, strcat('results/figures/', namedata), 'fig');
    saveas(h1, strcat('results/figures/', namedata), 'png');
end

if (save_data==1)
    %------------- Write all average regrets
    csv_out = [
        mean(reg_glmucb)
        mean(reg_gb)
        ]';
    
    header_row = 'GLM-UCB, GLM-GB';
    mycsvwrite(strcat('results/data/avg_regret_', namedata, '.csv'), ...
        csv_out, header_row, ',');
     
    %------------- Write raw dataall average regrets
    csvwrite(strcat('results/data/raw_regret_GLM_UCB_', namedata, '.csv'), ...
        reg_glmucb');
    csvwrite(strcat('results/data/raw_regret_GLM_GB_', namedata, '.csv'), ...
        reg_gb');
end

