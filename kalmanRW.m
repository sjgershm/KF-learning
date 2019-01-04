function [lik,latents] = kalmanRW(param,data)
    
    % Fit Kalman Rescorla-Wagner model to behavioral data.
    %
    % USAGE: [lik,latents] = KalmanRW(param,data)
    %
    % INPUTS:
    %   param - parameter vector with the following elements:
    %               param(1) = response noise variance (u)
    %               param(2) = response bias (b0)
    %               param(3) = response scaling (b1)
    %               param(4) = outcome noise variance
    %               param(5) = weight diffusion variance/volatility (optional)
    %   data - structure with the following fields:
    %               .X  = [N x D] matrix, where X(n,d) denotes stimulus
    %                     feature d on trial n
    %               .pred = [N x 1] vector of outcome predictions
    %               .r = [N x 1] vector of outcomes
    %
    % OUTPUTS:
    %   lik - log likelihood, assuming a Gaussian noise model
    %   latents - structure with the following fields:
    %               .modelpred = [N x 1] model predictions for behavioral responses
    %               .w = [N x D] associative weights
    %               .dt = [N x 1] prediction errors
    %               .rhat = [N x 1] outcome predictions
    %               .C = [N x D x D] posterior covariance (before updating)
    %               .K = [N x D] Kalman gain (learning rate)
    %               .conf = [N x 1] confidence / inverse precision
    %
    % Sam Gershman, January 2019
    
    % parameters
    u = param(1);       % response noise variance
    b0 = param(2);      % response bias
    b1 = param(3);      % response scaling
    s = param(4);       % outcome noise variance
    if length(param) > 4
        q = param(5);   % weight diffusion variance (volatility)
    else
        q = 0;          % default: diffusion variance is 0
    end
    
    % initialization
    [N,D] = size(data.X);   % # trials (N) and # features (D)
    w = zeros(D,1);         % weights
    C = eye(D);             % prior covariance (isotropic)
    Q = q*eye(D);           % diffusion covariance (isotropic)
    lik = 0;
    u = sqrt(u);
    
    % run Kalman filter
    for n = 1:N
        
        x = data.X(n,:);            % features on trial n
        rhat = x*w;                 % outcome prediction
        dt = data.r(n) - rhat;      % prediction error
        P = x*C*x'+s;               % marginal variance
        K = C*x'/P;                 % Kalman gain (learning rates)
        
        % likelihood
        modelpred = b0 + b1*rhat;
        lik = lik + lognormpdf(data.pred(n),modelpred,u);
        
        % store results
        if nargout > 1
            latents.modelpred(n,1) = modelpred;
            latents.conf(n,1) = 1./P;
            latents.w(n,:) = w;
            latents.C(n,:,:) = C;
            latents.K(n,:) = K;
            latents.dt(n,1) = dt;
            latents.rhat(n,1) = rhat;
        end
        
        % updates
        w = w + K*dt;               % weight update
        C = C + Q - K*x*C;          % posterior covariance update
        
    end