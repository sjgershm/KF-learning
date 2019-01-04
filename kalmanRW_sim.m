function data = kalmanRW_sim(param,data)
    
    % Generate simulated data from the Kalman Rescorla-Wagner model.
    %
    % USAGE: data = KalmanRW_sim(param,data)
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
    %               data.r = [N x 1] vector of outcomes
    %
    % OUTPUTS:
    %   data - same as input, but with an additional field (pred) storing
    %   the behavioral response
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
    
    % run Kalman filter
    for n = 1:N
        
        x = data.X(n,:);            % features on trial n
        rhat = x*w;                 % outcome prediction
        dt = data.r(n) - rhat;      % prediction error
        P = x*C*x'+s;               % marginal variance
        K = C*x'/P;                 % Kalman gain (learning rates)
        
        % response generation
        modelpred = b0 + b1*rhat;
        data.pred(n,1) = modelpred + normrnd(0,sqrt(u));
        
        % updates
        w = w + K*dt;               % weight update
        C = C + Q - K*x*C;          % posterior covariance update
        
    end