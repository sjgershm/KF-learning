function [results, bms_results] = fit_models(data,models,results)
    
    % Fit models to behavioral data. Three models are considered:
    %   1. Rescorla-Wagner
    %   2. Kalman Rescorla-Wagner with no volatility (q=0)
    %   3. Kalman Rescorla-Wagner with estimated volatility (q>0)
    %
    % USAGE: [results, bms_results] = fit_models(data,[models],[results])
    %
    % INPUTS:
    %   data - structure with the following fields:
    %               .X  = [N x D] matrix, where X(n,d) denotes stimulus
    %                     feature d on trial n
    %               .pred = [N x 1] vector of outcome predictions
    %               .r = [N x 1] vector of outcomes
    %   models (optional) - vector specifying which models to fit (default: all models)
    %   results (optional) - prior results structure to update
    %
    % OUTPUTS:
    %   results - structure containing results (see mfit_optimize.m)
    %   bms_results - model comparison results (see mfit_bms.m)
    %
    % Sam Gershman, January 2019
    
    likfuns = {'RW' 'kalmanRW' 'kalmanRW_q'};
    
    if nargin < 2; models = 1:length(likfuns); end
    
    % parameter bounds
    umin = 1e-3; umax = 10;
    b0min = -10; b0max = 10;
    b1min = -10; b1max = 10;
    smin = 1e-3; smax = 10;
    qmin = 0; qmax = 10;
    
    for mi = 1:length(models)
        m = models(mi);
        disp(['... fitting model ',num2str(m)]);
        fun = str2func(likfuns{m});
        
        switch likfuns{m}
            
            case 'RW'
                % Rescorla-Wagner
                
                param(1) = struct('name','u','logpdf',@(x) 0,'lb',umin,'ub',umax);
                param(2) = struct('name','b0','logpdf',@(x) 0,'lb',b0min,'ub',b0max);
                param(3) = struct('name','b1','logpdf',@(x) 0,'lb',b1min,'ub',b1max);
                param(4) = struct('name','lr','logpdf',@(x) 0,'lb',0,'ub',1);
                
            case 'kalmanRW'
                % Kalman Rescorla-Wagner
                
                param(1) = struct('name','u','logpdf',@(x) 0,'lb',umin,'ub',umax);
                param(2) = struct('name','b0','logpdf',@(x) 0,'lb',b0min,'ub',b0max);
                param(3) = struct('name','b1','logpdf',@(x) 0,'lb',b1min,'ub',b1max);
                param(4) = struct('name','s','logpdf',@(x) 0,'lb',smin,'ub',smax);
                
            case 'kalmanRW_q'
                % Kalman Rescorla-Wagner with non-zero diffusion variance (volatility)
                
                param(1) = struct('name','u','logpdf',@(x) 0,'lb',umin,'ub',umax);
                param(2) = struct('name','b0','logpdf',@(x) 0,'lb',b0min,'ub',b0max);
                param(3) = struct('name','b1','logpdf',@(x) 0,'lb',b1min,'ub',b1max);
                param(4) = struct('name','s','logpdf',@(x) 0,'lb',smin,'ub',smax);
                param(5) = struct('name','q','logpdf',@(x) 0,'lb',qmin,'ub',qmax);
                fun = str2func('kalmanRW');
                
        end
        
        results(m) = mfit_optimize(fun,param,data);
        clear param
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results,1);
    end