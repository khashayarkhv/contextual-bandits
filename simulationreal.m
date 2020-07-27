%% Simulations with real data.

% Setting dataset='EEG' generates Figure 4(a), dataset='EyeMovement'
% generates Figure 4(b), dataset='Cardiotocography' generates Figure 4(c),
% and finally dataset='Warfarin' generates Figure 4(d). 

%% Clear the workspace and close all figures.
close all;
clear;
random_seed = 10;
rng(random_seed);
tic
addpath('scripts');
addpath('configs');
addpath('datasets');
%
%% Dataset specifications.

% Note that the dataset should be one of 'Warfarin', 'EyeMovement', 'EEG',
% 'Cardiotocography'. Otherwise, the code will generate an error.

% NOTE: If you want to execute on other datasets, you need to modify this
% part and load your classification dataset. Note that you should load your
% data as data = [X; y], where X are contexts and y are correct labels.
% The covariates in X should be numerical and the labels in y should be 
% consecutive integers. 

dataset = 'EEG'; 

if(strcmp(dataset, 'Warfarin')==1)
    data = csvread('datasets/warfarin.csv', 1, 0);
elseif(strcmp(dataset, 'EyeMovement')==1)
    data = csvread('datasets/eye_movements.csv', 1, 0); 
elseif(strcmp(dataset, 'EEG')==1)
    data = csvread('datasets/eeg.csv', 1, 0);
elseif(strcmp(dataset, 'Cardiotocography')==1)
    data = csvread('datasets/cardiotocography.csv', 1, 0);
else
    if(ischar(dataset)==0)
      error('Error. Dataset should be a string.');
    end
    error('Error. \ndataset=%s is not among Warfarin, EyeMovement, EEG, or Cardiotocography.', ...
        dataset)
end

%% Data preprocessing: convert data into the desired format.

d = size(data,2)-1;

X = data(:, 1:d);
y = data(:, d+1);

T = size(X, 1);

k = max(y) - min(y)+1;
rewards = zeros(T,k);

for i=1:T
    X(i,:) = X(i, :) / norm(X(i, :));
    rewards(i, y(i) - min(y) + 1) = 1;
end

verbose = 0;   % Print results in each iteration or not.

save_figure = 0;  % Whether to save figures.
save_data = 0;    % Whether to save data. Note that the data is large. 


% Estimate sigma_e.
sigma_start = 1;
use_true_sigma_e = 0;
to_estimate_sigma_e = 1;
noise_input = 0;

% Dummy and unused parameters 
b = zeros(k, d);
sigma_e = 1;
sigma_x = 1;
xmax = 1;
%
%% Reading configs: load parameters. 
% NOTE: If you want to execute on other datasets, you need to modify this
% part and set your parameters. See config files for more information on
% how to do this.
if(strcmp(dataset, 'Warfarin')==1)
    config_warfarin;
elseif(strcmp(dataset, 'EyeMovement')==1)
    config_eyemovement; 
elseif(strcmp(dataset, 'EEG')==1)
    config_EEG;
else
    config_cardiotocography;
end

% If you think that simulations are taking too long, uncomment the 
% following line which decreases the number of Monte-Carlo simulations.
% ns = 10;
%
%% Matrices for saving Regrets.
reg_gb = zeros(ns, T);
reg_gf = zeros(ns, T);
reg_OFUL = zeros(ns, T);
reg_ols = zeros(ns, T);
reg_pd_ts = zeros(ns, T);
reg_oracle = zeros(ns,T);

if(strcmp(dataset, 'Warfarin')==1)
    reg_doctor = zeros(ns, T);
end

%% Matrices for saving Fraction of pulls.
frac_gb = zeros(ns, k);
frac_gf = zeros(ns, k);
frac_OFUL = zeros(ns, k);
frac_ols = zeros(ns, k);
frac_pd_ts = zeros(ns, k);
frac_oracle = zeros(ns, k);

%% Main code, run all algorithms.

gf_switch = zeros(ns, 1);  % Records whether Greedy-First switches or not.
range = 1:T;
logRange = log(range);

for s=1:ns
    per = randperm(size(X,1),T);
    X_per = X(per, :);
    rewards_per = rewards(per, :);
    
    [tmp_reg_ofls, tmp_frac_oful] = runOFUL(k, T, d, b, sigma_e, ...
        sigma_x, xmax, lambdaOFUL, deltaOFUL, ...
        sigma_start, use_true_sigma_e, to_estimate_sigma_e, verbose, ...
        X_per, noise_input, rewards_per);
    
    [tmp_reg_ols, tmp_frac_olsb, ~] = runOLSbandit(k, T, d, b, sigma_e, ...
        sigma_x, xmax, h, q, verbose, X_per, noise_input, rewards_per);
    
    [tmp_reg_gf, tmp_frac_gf, tmp_sw_t] = rungreedyfirst(k, T, d, b, ...
        sigma_e, sigma_x, xmax, h, q, t0, min_eig_threshold, ...
        random_initialization, verbose,  X_per, noise_input, rewards_per);
    
    [tmp_reg_gb, tmp_frac_gb] = rungreedybandit(k, T, d, b, sigma_e, ...
        sigma_x, xmax, random_initialization, verbose, X_per, ...
        noise_input, rewards_per);
    
    [tmp_reg_pd_ts, tmp_frac_pd_ts] = runpriordependentTS(k, T, d, b, ...
        sigma_e, sigma_x, xmax, prior_mean, prior_cov , sigma_start, ...
        use_true_sigma_e, to_estimate_sigma_e, verbose, X_per, ... 
        noise_input, rewards_per);
    
    [tmp_reg_oracle, tmp_frac_oracle] = runoracle(k, T, d, ...
        verbose, X_per, rewards_per);
    
    if(strcmp(dataset, 'Warfarin')==1)
        tmp_reg_doctor = rundoctor(T, rewards_per);
    end

    
    reg_oracle(s, :) = tmp_reg_oracle./range;
    if(strcmp(dataset, 'Warfarin')==1)
        reg_doctor(s, :) = tmp_reg_doctor./range;
    end
    reg_ols(s, :) = tmp_reg_ols./range;
    reg_OFUL(s, :) = tmp_reg_ofls./range;
    reg_gb(s, :) = tmp_reg_gb./range;
    reg_gf(s, :) = tmp_reg_gf./range;
    reg_pd_ts(s, :) = tmp_reg_pd_ts./range;
    
    %
    frac_oracle(s, :) = tmp_frac_oracle;
    frac_gf(s, :) = tmp_frac_gf;
    frac_gb(s, :) = tmp_frac_gb;
    frac_OFUL(s, :) = tmp_frac_oful;
    frac_ols(s, :) = tmp_frac_olsb;
    frac_pd_ts(s, :) = tmp_frac_pd_ts;
    
    gf_switch(s) = tmp_sw_t;
    
    fprintf('Round %d finished\n',s);
    toc
end
%
%% Save the results and create plots.
if(strcmp(dataset, 'Warfarin')==1)
    namedata = strcat('warfarin_ns_', num2str(ns));
elseif(strcmp(dataset, 'EyeMovement')==1)
    namedata = strcat('eyemovement_ns_', num2str(ns));  
elseif(strcmp(dataset, 'EEG')==1)
    namedata = strcat('EEG_ns_', num2str(ns));
else
    namedata = strcat('Cardiotocography_ns_', num2str(ns));
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

DarkPastelPurple = [0.5882    0.4353    0.8392];
DarkPastelGreen = [112 173 71] / 255;
PastelBrown = [0.5098    0.4118    0.3255];
PastelOrange = [237 125 49] / 255;
DarkPastelGrey = [120 120 120] / 255;
Gold = [214 163 0] / 255;
DarkBlue = [68 114 196] / 255;


A = shadedErrorBar(subint, reg_oracle(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'o-', 'Linewidth', 2, 'Color', DarkPastelPurple}, 1);
B = shadedErrorBar(subint, reg_gf(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'--', 'Linewidth', 2, 'Color', PastelOrange}, 1);
C = shadedErrorBar(subint, reg_OFUL(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'-.', 'Linewidth', 2, 'Color', DarkBlue}, 1);
D = shadedErrorBar(subint, reg_ols(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {':', 'Linewidth', 2, 'Color', DarkPastelGrey}, 1);
if(strcmp(dataset, 'Warfarin')==1)
    E = shadedErrorBar(subint, reg_doctor(:, subint), ...
        {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
        {'d-', 'Linewidth', 2, 'Color', PastelBrown}, 1);
end
G = shadedErrorBar(subint, reg_gb(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'Linewidth', 2, 'Color', DarkPastelGreen}, 1);
H = shadedErrorBar(subint, reg_pd_ts(:, subint), ...
    {@mean, @(x) 2*std(x)/sqrt(ns)}, ...
    {'x-', 'Linewidth', 2, 'Color', Gold}, 1);
if(strcmp(dataset, 'Warfarin')==1)
    l = legend([A.mainLine, B.mainLine, C.mainLine, D.mainLine, ...
        E.mainLine, G.mainLine, H.mainLine], ...
        'Oracle', 'GF', 'OFUL','OLS', 'Doctors', 'GB', 'Prior-Dependent TS');
else
    l = legend([A.mainLine, B.mainLine, C.mainLine, D.mainLine, ...
        G.mainLine, H.mainLine], ...
        'Oracle', 'GF', 'OFUL','OLS', 'GB', 'Prior-Dependent TS'); 
end

set(l, 'interpreter','latex')

set(gca,'FontSize', fontSize, 'FontName', fontName) 
ylabel('$\textrm{Fraction of incorrect decisions}$', 'Interpreter', ...
    'latex');
xlabel('$t$','Interpreter','latex');
grid on;

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
    if(strcmp(dataset, 'Warfarin')==1)
        csv_out = [
            mean(reg_gf)
            mean(reg_OFUL)
            mean(reg_ols)
            mean(reg_gb)
            mean(reg_pd_ts)
            mean(reg_oracle)
            mean(reg_doctor)
            ]';
            header_row = 'GF, OFUL, OLS, GB, PD-TS, Oracle, Doctors';
    else
        csv_out = [
            mean(reg_gf)
            mean(reg_OFUL)
            mean(reg_ols)
            mean(reg_gb)
            mean(reg_pd_ts)
            mean(reg_oracle)
            ]';
        header_row = 'GF, OFUL, OLS, GB, PD-TS, Oracle';
    end

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
    csvwrite(strcat('results/data/raw_regret_PD_TS_', namedata, '.csv'), ...
        reg_pd_ts');
    csvwrite(strcat('results/data/raw_regret_Oracle_', namedata, '.csv'), ...
        reg_oracle');
    if(strcmp(dataset, 'Warfarin')==1)
    csvwrite(strcat('results/data/raw_regret_Doctors_', namedata, '.csv'), ...
        reg_doctor');
    end
end
