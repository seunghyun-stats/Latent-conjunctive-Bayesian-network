%% latent conjunctive Bayesian network

%% parameter initialization
% diamond hierarchy with 8 attributes
K = 8;
adj_mat_true = zeros(K,K);
adj_mat_true(1, 2:3) = 1;
adj_mat_true(2:3, 4:6) = 1;
adj_mat_true(4:6, 7:8) = 1;

A_all = binary(0:(2^K-1), K);
A_true = get_patterns_from_hier(adj_mat_true);
[reach_mat_true, adj_mat_true] = get_reachability(A_true);

% proportion parameters
theta_posi_true = [0.9 0.8 0.8 0.7 0.7 0.7 0.6 0.6]';
[prop_true] = generate_prop_CBN(A_all, reach_mat_true, theta_posi_true);

nu_long = generate_prop_CBN(binary(0:(2^K-1), K), ...
    get_reachability(A_true), theta_posi_true);
nu_true = nonzeros(nu_long);

% item parameters
N = 500;
r = 0.1;

Q = [eye(K); eye(K); eye(K)];
for k = 1:(K-1)
    Q(k, k+1) = 1;
    Q(k+1, k) = 1;
    Q(K+k, k+1) = 1;
end

[J, K] = size(Q);

c_true = (1 - r) * ones(J,1); 
g_true = r * ones(J,1);


% simulate data and estimate using the 2-step EM algorithm
Nrep = 200;

A_cell = cell(Nrep,1);
M_pem = zeros(Nrep,1);
M_select = zeros(Nrep,1);
err_A_pem = ones(Nrep, 1);
err_A_select = ones(Nrep, 1);

mse_t = zeros(Nrep,1);
mse_nu = zeros(Nrep,1);
mse_c = zeros(Nrep,1);
mse_g = zeros(Nrep,1);
EBIC_lcbn = zeros(Nrep,1);
BIC_lcbn = zeros(Nrep,1);

mse_nu_pem = zeros(Nrep,1);
mse_c_pem = zeros(Nrep,1);
mse_g_pem = zeros(Nrep,1);
EBIC_pem = zeros(Nrep,1);
BIC_pem = zeros(Nrep,1);

iter_pem = zeros(Nrep,1);
iter_lcbn = zeros(Nrep,1);
iter_select = zeros(Nrep,1);
time_pem = zeros(Nrep,1);
time_lcbn = zeros(Nrep,1);
time_total = zeros(Nrep,1);
lambda_pem = zeros(Nrep,1);

nu_unif = ones(2^K,1)/2^K;
% nu_unif = nu_long;
% nu_unif = prop_true*0.4 + ones(2^K, 1) / 2^K*0.6; % initialization

rng(1)
parpool(4)
parfor (nn = 1:Nrep, 4)
    % initialize nu
    % nu_true = drchrnd(alpha, 1)';
    % nu_long = zeros(2^K, 1);
    % nu_long(bin2ind(A_true)+1) = nu_true;
    % prop_true = nu_long;

    % generate data
    [X, ~] = generate_X_DINA(N, prop_true, Q, c_true, g_true);
    
    tic

    %%%%%%%%% STEP 1: run the PEM algorithm %%%%%%%%%
    %%%%%%%%% PEM starts %%%%%%%%%%
    % lambda_vec = 0.8:-0.2:-4;
    lambda_vec = -0.4:-0.4:-4;
    tune_len = length(lambda_vec);
    EBIC_vec = zeros(tune_len, 1);
    size_vec = zeros(tune_len, 1);
    mat_nun = zeros(tune_len, 2^K);
    iter_vec = zeros(tune_len, 1);

    Z_candi = A_all;
    Z_candi_arr = cell(tune_len,1);
    c = c_true; g = g_true; nu = nu_unif; err_prob = 0.2;
    
    T_start = tic;
    for ii = 1:tune_len
        [nu, c, g, ~, iter_vec(ii)] = get_DINA_PEM(X, Q, A_all, lambda_vec(ii), c, g, nu, 0.01);
        mat_nun(ii, :) = nu';

        Z_candi = A_all(nu>0.5/N, :);
        Z_candi_arr{ii} = Z_candi;
        size_vec(ii) = size(Z_candi, 1);

        [nu_EM, ~, ~, loglik_EM_2] = get_DINA_EM(X, Q, Z_candi, err_prob);

        EBIC_vec(ii) = -2*loglik_EM_2 + ...
            (size_vec(ii) + 2*J) * log(N) + 2*1*log(nchoosek_prac(2*J + 2^K, 2*J + size_vec(ii) ));
    end


    % sum(mat_nun>0.5/N, 2)
    [~, Index] = min(EBIC_vec);
    tune_select = lambda_vec(Index);
    A_pem = Z_candi_arr{Index};
    
    % MSE in step 1
    [nu, c, g, loglik_EM, iter_select(nn)] = get_DINA_PEM(X, Q, A_all, tune_select, c_true, g_true, nu_long, 0.01);
    
    iter_pem(nn) = sum(iter_vec);
    time_pem(nn) = toc;
    lambda_pem(nn) = tune_select;
    
    A_cell{nn} = A_pem;
    mse_nu_pem(nn) = mean((nu_long - nu).^2);
    mse_c_pem(nn) = mean((c_true - c).^2);
    mse_g_pem(nn) = mean((g_true - g).^2);
    EBIC_pem(nn) = -2*loglik_EM + ...
        (size(A_pem, 1) + 2*J) * log(N) + 2*1*log(nchoosek_prac(2*J + 2^K, 2*J + size(A_pem, 1)));
    BIC_pem(nn) = -2*loglik_EM + (size(A_pem, 1) + 2*J) * log(N);

    %%%%%%%% STEP 2: Find hierarchy using A_pem %%%%%%%%
    [~, G] = get_reachability(A_pem);
    A_select = get_patterns_from_hier(G); % recovers the A_true even though A_pem is sparse

    M_pem(nn) = size(A_pem, 1);
    M_select(nn) = size(A_select, 1);

    err_A_pem(nn) = isequal(A_true, A_pem);
    err_A_select(nn) = isequal(A_true, A_select);

    %%%%%%%% STEP 3: Final EM using the estimated latent graph G %%%%%%%%%%
    tic
    [nu_est, theta_est, c_est, g_est, loglik, iter_lcbn(nn)] = get_CBN_DINA_EM(X, Q, A_select, c_true, g_true);
    time_lcbn(nn) = toc;

    nu_est_long = zeros(2^K, 1);
    nu_est_long(bin2ind(A_select)+1) = nu_est;

    mse_t(nn) = mean((theta_posi_true - theta_est).^2);
    mse_nu(nn) = mean((nu_long - nu_est_long).^2);
    mse_c(nn) = mean((c_true - c_est).^2);
    mse_g(nn) = mean((g_true - g_est).^2);

    EBIC_lcbn(nn) = -2*loglik + (K + 2*J) * log(N) + 2*1*log(nchoosek_prac(2*J + 2^K, 2*J + K ));
    BIC_lcbn(nn) = -2*loglik + (K + 2*J) * log(N);
    
    time_total(nn) = toc(T_start);
    fprintf('%d iter completed\n\n', nn);
end
save('DINA_CBN_500_0.1_200.mat')

%% evaluation
% [nu_long, nu_pem, nu_est_long]
index = find(mse_nu_pem);
index = find(err_A_select == 0)

mean(err_A_pem(index))
mean(err_A_select(index))

sqrt(mean(mse_nu_pem(index)))
sqrt(mean([mse_g_pem(index); mse_c_pem(index)]))

sqrt(mean(mse_t(index) ))  % not possible for simulation 2
sqrt(mean(mse_nu(index))) % appx half smaller compared to pem
sqrt(mean([mse_g(index); mse_c(index)])) % similar to PEM

[BIC_pem, BIC_lcbn]
mean(EBIC_pem(index) > EBIC_lcbn(index))
median(BIC_pem(index) > BIC_lcbn(index))

mean(iter_pem)
mean(iter_lcbn)
mean(iter_select)
mean(time_pem)
mean(time_lcbn)
mean(time_total)
mean(lambda_pem)
