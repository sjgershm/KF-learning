clear

% create some fake data, generated from the Kalman Rescorla-Wagner model with no volatility
S = 5;          % number of subjects
N = 100;        % number of trials
X = ones(N,1);  % constant stimulus
r = ones(N,1);  % constant outcome

for s = 1:S
    param = [0.01 0 1 rand];    % random outcome noise variance for each subject
    noise(s,1) = param(4);
    data(s) = kalmanRW_sim(param,struct('X',X,'r',r,'N',N));
end

% fit data
[results, bms_results] = fit_models(data);

% plot results
figure;

subplot(1,3,1);
scatter(noise,results(2).x(:,4)); lsline
xlabel('True outcome noise','FontSize',20);
ylabel('Estimated outcome noise','FontSize',20);

subplot(1,3,2);
plot([data(1).pred results(2).latents(1).modelpred],'LineWidth',4);
legend({'Data' 'Model'},'FontSize',20,'Location','Best');
xlabel('Trial','FontSize',20);
ylabel('Outcome prediction','FontSize',20);

subplot(1,3,3);
barh(bms_results.pxp);
set(gca,'YTickLabel',{'RW' 'Kalman RW (q=0)' 'Kalman RW (q>0)'},'FontSize',20);
xlabel('PXP','FontSize',20);

set(gcf,'Position',[200 200 1200 400]);