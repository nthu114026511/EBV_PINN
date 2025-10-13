%% =========================================================
%  RK4 with fraction (impulse) jumps under LQ model
%  Continuous ODE between jumps:
%    B' = r*B*(1 - B/K)
%    R' = sigma0*B - k12*R
%    E' = k12*R - kc*E
%  Jump at t = t_j with dose d_j:
%    SF = exp(-alpha*d_j - beta*d_j^2)
%    B^+ = B^- * SF
%    R^+ = R^- + sigma1 * (B^- - B^+)
%    E^+ = E^- (continuous)
% =========================================================
clear; clc; close all;

%% ---------- Parameters & ICs ----------
r      = 0.10;
K      = 1.00;
sigma0 = 0.30;
sigma1 = 1.00;     % 被殺的 B 立即轉成 R 的比例
k12    = 0.20;
kc     = 0.15;

B0 = 0.05; R0 = 0.00; E0 = 0.00;

%% ---------- LQ parameters & fraction schedule ----------
alpha  = 0.30;     % 腫瘤 alpha (1/Gy)
beta   = 0.03;     % 腫瘤 beta  (1/Gy^2)

% 3 個跳躍點（可自行調整）
%等劑量、指定時刻：
t_frac = [5; 12; 20];   % 跳躍時刻
d_frac = [2;  2;  2];   % 對應劑量 (Gy)
%不同劑量（例如最後一次 boost）：
%t_frac = [5; 12; 20];
%d_frac = [2;  2;  4];
%等間隔三次、每天一照：
%t_frac = (1:3).';
%d_frac = 2*ones(3,1);


%% ---------- Global time grid ----------
t0 = 0;  tf = 60;
h  = 0.1;               % 基本步長；事件前會自動縮步以精確對齊 t_j

%% ---------- Run piecewise RK4 with jumps ----------
pars = struct('r',r,'K',K,'sigma0',sigma0,'k12',k12,'kc',kc);

[T, Y, Tjump, JumpInfo] = rk4_with_fractions(@rhs, [t0 tf], [B0; R0; E0], h, ...
                              t_frac, d_frac, alpha, beta, sigma1, pars);

B = Y(:,1); R = Y(:,2); E = Y(:,3);

%% ---------- Plot ----------
figure; 
subplot(3,1,1); plot(T,B,'LineWidth',1.5); ylabel('B');
hold on; xline(Tjump,'k:'); title('B, R, E with 3 fraction jumps (LQ)');
subplot(3,1,2); plot(T,R,'LineWidth',1.5); ylabel('R'); hold on; xline(Tjump,'k:');
subplot(3,1,3); plot(T,E,'LineWidth',1.5); ylabel('E'); xlabel('t'); hold on; xline(Tjump,'k:');

%% ---------- 檢視每次分次的 SF 與跳躍量（可選） ----------
disp(struct2table(JumpInfo));

%% ---------- 儲存數據供 Python PINN 訓練使用 ----------
% 建立輸出資料夾
output_dir = '../only_ode/data';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 方法 1: 儲存為 CSV 檔（推薦用於 Python）
data_table = table(T, B, R, E, 'VariableNames', {'t', 'B', 'R', 'E'});
csv_filename = fullfile(output_dir, 'evb_training_data.csv');
writetable(data_table, csv_filename);
fprintf('✓ CSV 數據已儲存至: %s\n', csv_filename);

% 方法 2: 儲存為 MAT 檔（含完整資訊）
mat_filename = fullfile(output_dir, 'evb_training_data.mat');
save(mat_filename, 'T', 'B', 'R', 'E', 'Tjump', 'JumpInfo', ...
     'r', 'K', 'sigma0', 'sigma1', 'k12', 'kc', 'alpha', 'beta', ...
     't_frac', 'd_frac');
fprintf('✓ MAT 數據已儲存至: %s\n', mat_filename);

% 方法 3: 儲存跳躍點資訊為 CSV（用於標記特殊時刻）
if ~isempty(JumpInfo)
    jump_table = struct2table(JumpInfo);
    jump_csv = fullfile(output_dir, 'evb_jump_info.csv');
    writetable(jump_table, jump_csv);
    fprintf('✓ 跳躍資訊已儲存至: %s\n', jump_csv);
end

% 方法 4: 儲存參數為 JSON 格式（方便 Python 讀取）
params = struct('r', r, 'K', K, 'sigma0', sigma0, 'sigma1', sigma1, ...
                'k12', k12, 'kc', kc, 'alpha', alpha, 'beta', beta, ...
                't_frac', t_frac, 'd_frac', d_frac, ...
                't0', t0, 'tf', tf, 'h', h);
json_filename = fullfile(output_dir, 'evb_parameters.json');
json_str = jsonencode(params, 'PrettyPrint', true);
fid = fopen(json_filename, 'w');
fprintf(fid, '%s', json_str);
fclose(fid);
fprintf('✓ 參數已儲存至: %s\n', json_filename);

fprintf('\n數據維度: %d 個時間點 × 4 個變數 (t, B, R, E)\n', length(T));
fprintf('時間範圍: [%.2f, %.2f]\n', min(T), max(T));
fprintf('跳躍點數: %d\n', length(Tjump));

%% ===================== Subfunctions =====================

function dy = rhs(~, y, p)
    B = y(1); R = y(2); E = y(3);
    dB = p.r*B*(1 - B/p.K);
    dR = p.sigma0*B - p.k12*R;
    dE = p.k12*R - p.kc*E;
    dy = [dB; dR; dE];
end

function [T, Y, Tjump, JumpInfo] = rk4_with_fractions(f, tspan, y0, h, ...
                                                      t_frac, d_frac, alpha, beta, sigma1, p)
    % 準備事件（只取位於 (t0, tf) 之內的）
    t0 = tspan(1); tf = tspan(2);
    mask = (t_frac > t0) & (t_frac < tf);
    t_events = t_frac(mask);
    d_events = d_frac(mask);
    [t_events, idx] = sort(t_events);      % 依時間排序
    d_events = d_events(idx);
    nEvents = numel(t_events);

    % 預留輸出容器
    T = t0;  Y = y0.';        % 行向量記錄
    Tjump = zeros(nEvents,1);
    JumpInfo = struct('t',{},'dose',{},'SF',{},'dB_instant',{},'R_gain',{}); %#ok<AGROW>

    t = t0;  y = y0;

    % 每一段： [t, next_event] 之間用 RK4；到點後施加跳躍
    for j = 1:nEvents+1
        if j<=nEvents, tEnd = t_events(j); else, tEnd = tf; end

        % --- integrate with RK4 to hit tEnd exactly ---
        while t < tEnd - 1e-14
            hstep = min(h, tEnd - t);
            y = rk4_step(f, t, y, hstep, p);
            t = t + hstep;
            T(end+1,1) = t;             %#ok<AGROW>
            Y(end+1,:) = y.';           %#ok<AGROW>
        end

        % --- apply jump at event j ---
        if j<=nEvents
            Bm = y(1); Rm = y(2); Em = y(3); % minus state
            dose = d_events(j);
            SF   = exp(-alpha*dose - beta*dose^2);
            Bp   = Bm * SF;
            dBinst = Bm - Bp;                  % 立即被殺死的量
            Rp   = Rm + sigma1 * dBinst;       % 立即注入到 R
            Ep   = Em;                         % 連續

            y = [Bp; Rp; Ep];                  % replace state

            T(end+1,1) = t;                    %#ok<AGROW>
            Y(end+1,:) = y.';                  %#ok<AGROW>

            Tjump(j) = t;
            JumpInfo(j).t = t;
            JumpInfo(j).dose = dose;
            JumpInfo(j).SF = SF;
            JumpInfo(j).dB_instant = dBinst;
            JumpInfo(j).R_gain = sigma1 * dBinst;
        end
    end
end

function y1 = rk4_step(f, t, y, h, p)
    k1 = f(t,          y,              p);
    k2 = f(t + h/2.0,  y + h*k1/2.0,   p);
    k3 = f(t + h/2.0,  y + h*k2/2.0,   p);
    k4 = f(t + h,      y + h*k3,       p);
    y1 = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4);
    % 保正 (可選)：避免數值誤差變成負
    y1 = max(y1, 0);
end