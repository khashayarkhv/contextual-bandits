%% Simulations with synthetic and linear rewards. 

% Setting correct_noise = 1 and intercept = 0. generates Figure 2(a).
% Setting correct_noise = 0 and intercept = 0. generates Figure 2(b).
% Setting correct_noise = 1 and intercept = 1. generates Figure 2(c).
% Setting correct_noise = 0 and intercept = 1. generates Figure 2(d).
%
%% Clear the workspace and close all figures.
close all;
clear;
random_seed = 10;
rng(random_seed);
tic
addpath('scripts');
%
%% CHANGE THESE BINARY INPUTS TO GENERATE FIGURES 2(a) TO 2(d).
correct_noise = 0;  % Binary, OFUL and TS use correct priors/noise.
intercept = 1;    % Binary, whether to include intercept or not.

%% Problem parameters.
% Number of simulation runs. 1000 simulations takes a long time to run, but
% this is necessary when intercept=1 (as the "bad" event for greedy where
% it drops an arm happens with a very small probability). If you think 
% that simulations are taking too long, reduce it to ns = 100 (less
% accurate).

ns = 1000;   
k = 2;  % Number of Arms
T = 1e4;  % Length of each simulation
dim_disc = 0;  % Dimension of binary distribution
dim_cont = 3;  % Dimension of continuous distribution
verbose = 0;  % Print results in each iteration or not.

save_figure = 0; % Whether to save figures.
save_data = 0;    % Whether to save data. Note that the data is large. 

if(intercept==1)
    dim_cont = dim_cont + 1;  % Increase continuous dimension.
end

d = dim_cont + dim_disc;   % Total dimension.

%% Parameter of algorithms.
% OLS bandit Parameters.
q = 1;
h = 5;
% OFUL parameters.
lambdaOFUL = 1;
deltaOFUL = 0.99;
% Prior-Free Thompson Sampling parameters.
deltaTS = 0.99;
priorTS = 0.2;
% Greedy-First parameters.
min_eig_threshold = 1e-5;
t0 = 4 * k * d;
sigma_start = 100;
% Number of rounds of random sampling in the beginning (for Greedy and
% Greedy-First)
random_initialization = 0;
%

%% Context and noise specifications.
sigma_e = 0.5; % Noise Variance(we assume gaussian)

cont_uplim = 1;   % Gaussian upper truncation limit.
cont_lowlim = -1;   % Gaussian lower truncation limit.
sigma_x = 0.5 * eye(dim_cont);   % Covariance matrix of gaussian contexts.

disc_uplim = 1;   % Discrete (Rademacher) upper limit.
disc_lowlim = -1;   % Discrete (Rademacher) lower limit.
disc_lowlim_prob = 0.5;   % Probability of lower limit for discrete.

noise_input = 1;

xmax = max([abs(disc_lowlim), abs(disc_uplim), ...
    abs(cont_lowlim), abs(cont_uplim)]);    % Maximum l_infinity norm.

intercept_scale = 2e-1 * intercept;   % The intercept value.

%% Matrices for saving regrets.
reg_gb = zeros(ns, T);
reg_gf = zeros(ns, T);
reg_OFUL = zeros(ns, T);
reg_ols = zeros(ns, T);
reg_pf_ts = zeros(ns, T);
reg_pd_ts = zeros(ns, T);

%% Matrices for saving fraction of pulls.
frac_gb = zeros(ns, k);
frac_gf = zeros(ns, k);
frac_OFUL = zeros(ns, k);
frac_olsb = zeros(ns, k);
frac_pf_ts = zeros(ns, k);
frac_pd_ts = zeros(ns, k);

%% Main code, run all algorithms.

gf_switch = zeros(ns, 1);  % Records whether Greedy-First switches or not.
range = 1:T;

for s=1:ns
    % Generate arm parameters.
    if(correct_noise==1)
        b = randn(k,d);
        prior_mean = zeros(d,1);
        prior_cov = eye(d,d);
        use_true_sigma_e = 1;
        to_estimate_sigma_e = 0;
    else   % Mixture model.
        ber = rand;
        if (ber>.5)
            b = mvnrnd(-ones(d,1), eye(d,d), k) / 2;
        else
            b = mvnrnd(ones(d,1), eye(d,d), k) / 2;
        end 
        use_true_sigma_e = 0;
        to_estimate_sigma_e = 1;
        prior_mean = zeros(d,1);
        prior_cov = 100 * eye(d,d);
    end

    fprintf('Round %d started. \n', s);
    % Generate continuous covariates from Gaussian distribution.
    Xcont = max(cont_lowlim, min(cont_uplim, ...
        mvnrnd(zeros(dim_cont ,1), sigma_x, T)));
    
    % Generate continuous covariates from Rademacher distribution.
    Xdisc = (disc_uplim - disc_lowlim) * ...
        (rand(T, dim_disc) < disc_lowlim_prob) + disc_lowlim;
    % Generate Gaussian noise.
    e = randn(T,1)*sigma_e;
    
    X=[Xcont, Xdisc];
    if (intercept ==1)
        X(:,1) = ones(T, 1);    % Ignore the first dimension.  
        b(:,1)= b(:, 1)*intercept_scale;  % Scale of intercept.
    end
    [tmp_reg_ofls, tmp_frac_oful] = runOFUL(k, T, d, b, sigma_e, ...
        sigma_x, xmax, lambdaOFUL, deltaOFUL, ...
        sigma_start, use_true_sigma_e, to_estimate_sigma_e, verbose, ...
        X, noise_input, e);
    [tmp_reg_ols, tmp_frac_olsb] = runOLSbandit(k, T, d, b, sigma_e, ...
        sigma_x, xmax, h, q, verbose, X, noise_input, e);
    [tmp_reg_gf, tmp_frac_gf, tmp_sw_t] = rungreedyfirst(k, T, d, b, ...
        sigma_e, sigma_x, xmax, h, q, t0, min_eig_threshold, ...
        random_initialization, verbose,  X, noise_input, e);
    [tmp_reg_gb, tmp_frac_gb] = rungreedybandit(k, T, d, b, sigma_e, ...
        sigma_x, xmax, random_initialization, verbose, X, noise_input, e);
    [tmp_reg_pf_ts, tmp_frac_pf_ts] = runpriorfreeTS(k, T, d, b, sigma_e, ...
        sigma_x, xmax, deltaTS, priorTS, sigma_start, use_true_sigma_e, ...
        to_estimate_sigma_e, verbose, X, noise_input, e);
    [tmp_reg_pd_ts, tmp_frac_pd_ts] = runpriordependentTS(k, T, d, b, ...
        sigma_e, sigma_x, xmax, prior_mean, prior_cov , sigma_start, ...
        use_true_sigma_e, to_estimate_sigma_e, verbose, X, noise_input, e);
    
    reg_OFUL(s,:) = tmp_reg_ofls;
    reg_ols(s,:) = tmp_reg_ols;
    reg_gb(s,:)=tmp_reg_gb;
    reg_pf_ts(s,:)=tmp_reg_pf_ts;
    reg_gf(s,:)=tmp_reg_gf;
    reg_pd_ts(s,:) = tmp_reg_pd_ts;
    %
    fprintf('Total Regret after %d simulations \n', s)
    fprintf('GB=%f \n', mean(reg_gb(1:s,end)))
    fprintf('GF=%f \n', mean(reg_gf(1:s,end)))
    fprintf('OFUL=%f \n', mean(reg_OFUL(1:s,end)))
    fprintf('OLS=%f \n', mean(reg_ols(1:s,end)))
    fprintf('PF-TS=%f \n', mean(reg_pf_ts(1:s,end)))
    fprintf('PD-TS=%f \n', mean(reg_pd_ts(1:s,end)))
    
    %
    frac_gf(s,:) = tmp_frac_gf;
    frac_gb(s,:) = tmp_frac_gb;
    frac_OFUL(s,:) = tmp_frac_oful;
    frac_olsb(s,:) = tmp_frac_olsb;
    frac_pf_ts(s,:) = tmp_frac_pf_ts;
    frac_pd_ts(s,:) = tmp_frac_pd_ts;
    gf_switch(s) = tmp_sw_t;
    toc
end

%% Save the results and create plots.

% Generate file name.
if(correct_noise==1)
    if(intercept==0)
        namedata = strcat('Synth_TruePrior_CovDiv_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_disdim_', num2str(dim_disc), '_contdim_', ...
            num2str(dim_cont),'_noise_', strrep(num2str(sigma_e),'.','_'));
    else
        namedata = strcat('Synth_TruePrior_NoCovDiv_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_disdim_', num2str(dim_disc), '_contdim_', ...
            num2str(dim_cont),'_noise_', strrep(num2str(sigma_e),'.','_'));
    end
else
    if(intercept==0)
        namedata = strcat('Synth_IncorrectPrior_CovDiv_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_disdim_', num2str(dim_disc), '_contdim_', ...
            num2str(dim_cont),'_noise_', strrep(num2str(sigma_e),'.','_'));
    else
        namedata = strcat('Synth_IncorrectPrior_NoCovDiv_seed_', ...
            num2str(random_seed), '_ns_', num2str(ns), ...
            '_T_', num2str(T),'_k_', num2str(k), ...
            '_disdim_', num2str(dim_disc), '_contdim_', ...
            num2str(dim_cont),'_noise_', strrep(num2str(sigma_e),'.','_')); 
    end
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

DarkPastelRed = [0.7608    0.2314    0.1333];
DarkPastelBlue = [0.4667    0.6196    0.7961];
PastelRed = [1.0000    0.4118    0.3804];
DarkPastelPurple = [0.5882    0.4353    0.8392];
DarkPastelGreen = [112 173 71]/255;
PastelBrown = [0.5098    0.4118    0.3255];
PastelOrange = [237 125 49]/255;
DarkPastelGrey = [120 120 120]/255;
Gold = [214 163 0]/255;
DarkBlue = [68 114 196]/255;
LighterBlue = [91 155 213]/255;

B=shadedErrorBar(subint, reg_gf(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'--', 'Linewidth', 2, 'Color',PastelOrange}, 1);
C=shadedErrorBar(subint, reg_OFUL(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'-.', 'Linewidth', 2, 'Color', DarkBlue}, 1);
D=shadedErrorBar(subint, reg_ols(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {':', 'Linewidth', 2, 'Color', DarkPastelGrey}, 1);
E=shadedErrorBar(subint, reg_gb(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'Linewidth', 2, 'Color', DarkPastelGreen}, 1);
F=shadedErrorBar(subint, reg_pf_ts(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'x-', 'Linewidth', 2, 'Color',Gold}, 1);
G=shadedErrorBar(subint, reg_pd_ts(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'*-', 'Linewidth', 2, 'Color', DarkPastelRed}, 1);


l = legend([B.mainLine, C.mainLine,D.mainLine, E.mainLine, ...
    F.mainLine, G.mainLine], 'GF', 'OFUL', 'OLS', 'GB', ... 
    'Prior-free TS', 'Prior-dependent TS', 'Location', 'northeast');

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
        mean(reg_gf)
        mean(reg_OFUL)
        mean(reg_ols)
        mean(reg_gb)
        mean(reg_pf_ts)
        mean(reg_pd_ts)
        ]';
    
    header_row = 'GF, OFUL, OLS, GB, PF-TS, PD-TS';
    mycsvwrite(strcat('results/data/avg_regret_', namedata, '.csv'), ...
        csv_out, header_row, ',');
     
    %------------- Write raw dataall average regrets
    csvwrite(strcat('results/data/raw_regret_GF_', namedata, '.csv'), ...
        reg_gf');
    csvwrite(strcat('results/data/raw_regret_OFUL_', namedata, '.csv'), ...
        reg_OFUL');
    csvwrite(strcat('results/data/raw_regret_OLS_', namedata, '.csv'), ...
        reg_ols');
    csvwrite(strcat('results/data/raw_regret_GB_', namedata, '.csv'), ...
        reg_gb');
    csvwrite(strcat('results/data/raw_regret_PF_TS_', namedata, '.csv'), ... 
        reg_pf_ts');
    csvwrite(strcat('results/data/raw_regret_PD_TS_', namedata, '.csv'), ...
        reg_pd_ts');
end
