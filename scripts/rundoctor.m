%% rundoctor.m

% Trains the model of doctors on Warfarin dataset.

% This function trains the doctors model on Warfarin dataset. In
% particular, the doctors always assign the patients to the medium dose in
% (ind = 1) in their first visit.

%% Inputs:
%   T: Time horizon.
%   rewards: A T*k matrix, containing the rewards of all actions at all
%   time periods.
%
%% Outputs:
%
%   regret: Cumulative regret as a running sum over regret terms.
%

function regret = rundoctor(T, rewards)

regret = zeros(1,T);

for i=1:T 
    bestreward = max(rewards(i,:));
    ourreward = rewards(i,2);
    
    if (i==1)
        regret(i) = bestreward - ourreward;
    else
        regret(i) = regret(i-1) + bestreward - ourreward;
    end

end

