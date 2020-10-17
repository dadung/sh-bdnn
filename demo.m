clear;

addpath(genpath('DeepNN_HASH'));
addpath(genpath('utils'));
addpath(genpath('minFunc'));
addpath(genpath('yael'));
addpath(genpath('DiscretHashing-master'));

%% CONFIGURE DATASET FOR EXPERIMENT
dataset = 'cifar'; % please change 'mnist' or 'cifar-10'
dimen = 800;
basedir = ['dataset/' dataset '/'];
bit_evals = [8 16 24 32]; % number of bits you want to evaluate

%% LET'S GO
map_our_hash = [];
map_sdh = [];
disp('============== RUNNING BINARY DEEP NEURAL NETWORK ===============');

for bit = bit_evals
    fprintf(2,'USING %d BITS\n', bit);
    map = runHash( dataset, basedir, bit, dimen );
    map_our_hash = [map_our_hash ; map];
end

disp('================ RUNNING SDH (CVPR 2015) FOR COMPARISON ======================');
for bit = bit_evals
    map = runSDH( dataset, basedir, bit, dimen );
    map_sdh = [map_sdh ; map];
end

%% VISUALIZATION
interval = [0 8 16 24];
hold on;
font = 15;
set(gca, 'FontSize', font);
xlim([0 24]);
% ylim([20 80]);
plot(interval, map_our_hash, 'Color', 'red', 'Marker', 's', 'Linewidth', 2);
hold on;
plot(interval, map_sdh, 'Color', 'blue', 'Marker', 'o', 'Linewidth', 2);

lngd = legend('SH-BDNN', 'SDH (CVPR15)');
set(lngd, 'Location', 'southeast');
set(lngd, 'interpreter', 'latex');
set(lngd, 'fontsize', 15);
grid on;
xlabel('number of bits', 'fontsize', 20);
ylabel('mAP','fontsize', 20);
set(gca,'XTick',0:8:24);
set(gca,'XTickLabel',{'8','16','24','32'});
set(gca,'YTick',20:10:70);