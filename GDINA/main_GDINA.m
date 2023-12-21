%% latent conjunctive Bayesian network - GDINA

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
prop_true = generate_prop_CBN(A_all, reach_mat_true, theta_posi_true);

% item parameters
n_in = size(A_true, 1);

Q = [eye(K); eye(K); eye(K)];
for k = 1:(K-1)
    Q(k, k+1) = 1;
    Q(k+1, k) = 1;
    Q(K+k, k+1) = 1;
end

[J, K] = size(Q);

g = 0.1; % change accordingly
c = 0.9;

S = zeros(J, 1);
T = zeros(J, 1);
theta_true = ones(J, n_in) * g;
delta_true = zeros(J, n_in);

attr_combo = A_true(2:n_in, 1:K);
expand_A = [ones(size(A_true,1), 1), prod(bsxfun(@power, reshape(A_true, [size(A_true,1) 1 K]), ...
    reshape(attr_combo, [1 n_in - 1 K])), 3)];

expand_Q = zeros(J, n_in);
for j = 1:J
    k_index = nonzeros((1:K) .* (Q(j, :) == 1))';
    sub_attributes = unique(A_true(:, k_index), 'rows', 'stable');

    for jj = 1:length(sub_attributes)
        n_index = find(ismember(A_true(:, k_index), sub_attributes(jj, :), 'rows'), 1);
        expand_Q(j, n_index) = 1;
    end
end
for j = 1:J
    k_index = nonzeros((1:n_in) .* expand_Q(j,:));
    unique_rows = unique(expand_A(:, k_index), 'rows');
    S(j) = length(unique_rows); % K_j, number of effective DINA parameters for item j
    if S(j) == 1
        theta_true(j, 1) = g;
    else
        for jj = 1:S(j)
            ind_jj = find(ismember(expand_A(:, k_index), unique_rows(jj, :), 'rows'));
            theta_true(j, ind_jj) = ones(length(ind_jj), 1)' * (g + (c-g)/(S(j) - 1)*...
                (sum(unique_rows(jj, :))-1));
        end
    end
end

for j = 1:J
    delta_true(j, : ) = expand_A \ theta_true(j, :)';
    T(j) = size(nonzeros(delta_true(j, : )), 1);
end
JJ = sum(T);

nu_true = nonzeros(generate_prop_CBN(binary(0:(2^K-1), K), ...
    get_reachability(A_true), theta_posi_true));

ind = get_ind_from_hier(adj_mat_true);

% convert theta, delta_true into J*2^K long matrix
delta_in = zeros(J, 2^K);
delta_in(:,ind) = delta_true;

theta_in = zeros(J, 2^K);
attr_combo_all = A_all(2:2^K, 1:K);
expand_A_all = [ones(size(A_all,1), 1), prod(bsxfun(@power, reshape(A_all, [size(A_all,1) 1 K]), ...
    reshape(attr_combo_all, [1 2^K- 1 K])), 3)];
for j = 1:J
    theta_in(j, :) = expand_A_all * delta_in(j,:)';
end

% simulate data and estimate using the 2-step EM algorithm
N = 500;
Nrep = 100;

M_pem = zeros(Nrep,1);
M_select = zeros(Nrep,1);
err_A_pem = ones(Nrep, 1);
err_A_select = ones(Nrep, 1);

mse_t = zeros(Nrep,1);
mse_nu = zeros(Nrep,1);
mse_delta = zeros(Nrep,1);
loglik_lcbn = zeros(Nrep,1);

mse_nu_pem = zeros(Nrep,1);
mse_delta_pem = zeros(Nrep,1);
loglik_pem = zeros(Nrep,1);

EBIC_pem = zeros(Nrep,1);
EBIC_lcbn = zeros(Nrep,1);

lambda_vec = 0.4:-0.4:-3;
tune_len = length(lambda_vec);
EBIC_vec = zeros(tune_len, 1);
size_vec = zeros(tune_len, 1);
mat_nun = zeros(tune_len, 2^K);

nu_in = ones(2^K, 1) / 2^K;

rng(2023);
parpool(4)
parfor (nn = 1:Nrep, 4)
    % generate data
    % rng(2022+nn);
    [X, ~] = generate_X_GDINA(N, nu_true, Q, delta_true, A_true);

    %%% STEP 1: run the PEM algorithm 
    %%%%%%%%% PEM starts %%%%%%%%%%
    
    lambda_vec = -1.2:-0.4:-4.0; 
    tune_len = length(lambda_vec);
    EBIC_vec = zeros(tune_len, 1);
    size_vec = zeros(tune_len, 1);
    mat_nun = zeros(tune_len, 2^K);
    Z_candi = A_all;
    Z_candi_arr = cell(tune_len,1);
    
    c_true = c; g_true = g;
    
    for ii = 1:tune_len
        [nu, theta, delta, loglik] = get_GDINA_PEM(X, Q, A_all, nu_in, lambda_vec(ii), theta_in, 0.01);

        mat_nun(ii, :) = nu';
        Z_candi = A_all(nu>0.5/N, :); % selected A
        Z_candi_arr{ii} = Z_candi;
        size_vec(ii) = size(Z_candi, 1);
        
        if size_vec(ii) == 1
            break
        end
    
    
        EBIC_vec(ii) = -2*loglik + ...
             size_vec(ii) * log(N) + 2*1*log(nchoosek_prac( 2^K, size_vec(ii) ));
    
        % fprintf('lambda %1.2f completed\n\n', lambda_vec(ii));
    end

    % sum(mat_nun>0.5/N, 2)
    [~, Index] = min(EBIC_vec);
    tune_select = lambda_vec(Index);
    A_pem = Z_candi_arr{Index};
    
    [nu, theta, delta_est, loglik] = get_GDINA_PEM(X, Q, A_all, nu, lambda_vec(Index), theta_in, 0.01);
    
    mse_nu_pem(nn) = mean((prop_true - nu).^2);
    if isequal(A_true, A_pem)
        [~, ~, delta_est, ~] = get_GDINA_PEM(X, Q, A_true, ones(n_in, 1) / n_in, lambda_vec(Index), theta_true, 0.01);
        mse_delta_pem(nn) = mean(nonzeros((delta_true - delta_est).^2));
    end
    
    mse_theta_pem(nn) = mean(mean((theta_true - theta(:, ind)).^2));
    EBIC_pem(nn) = -2*loglik + ...
        	(size(A_pem, 1) + JJ) * log(N) + 2*1*log(nchoosek_prac(JJ + 2^K, JJ + size(A_pem, 1)));

    %%%%%%%% STEP 2: Find hierarchy using A_pem %%%%%%%%
    [~, G] = get_reachability(A_pem);
    A_select = get_patterns_from_hier(G);

    M_pem(nn) = size(A_pem, 1);
    M_select(nn) = size(A_select, 1);
    err_A_pem(nn) = isequal(A_true, A_pem);
    err_A_select(nn) = isequal(A_true, A_select);
    
    %%%%%%%% STEP 3: Final EM using the estimated latent graph G %%%%%%%%%%
    [t_est, nu_est, theta_est, delta_est, loglik] = get_CBN_GDINA_EM(X, Q, A_select, G);

    mse_t(nn) = mean((theta_posi_true - t_est).^2);
    
    if isequal(A_select, A_true)
        mse_nu(nn) = sum((nu_true - nu_est).^2) / 2^K;
        mse_delta(nn) = mean(nonzeros((delta_true - delta_est).^2));
        % mse_theta(nn) = mean(mean((theta_true - theta_est).^2));
    end
    
    EBIC_lcbn(nn) = -2*loglik + (K + JJ) * log(N) + 2*1*log(nchoosek_prac(JJ + 2^K, JJ + K ));

    loglik_lcbn(nn) = loglik;
    fprintf('%d iter completed\n\n', nn);
end


mean(M_pem(1:80) == 15)
mean(M_select(1:80) == 15)

mean(err_A_pem)
mean(err_A_select(1:u))

u = Nrep;
sqrt(mean(mse_nu_pem(1:u)))
sqrt(mean(nonzeros(mse_delta_pem)))

sqrt(mean(mse_t(1:u)))
sqrt(mean(nonzeros(mse_nu(1:u))))
sqrt(mean(nonzeros(mse_delta)))

mean(EBIC_pem > EBIC_lcbn)
