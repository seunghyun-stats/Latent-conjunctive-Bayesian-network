function [nu, theta, c, g, loglik, n_iter] = get_CBN_DINA_EM(X, Q, A_in, c_in, g_in)
%%% work with theta instead of nu
%%% A_in -> alpha's that respect the partial order
%%% G = adj_mat_true

%%%% definitions
[J, K] = size(Q);

N = size(X, 1);
resp_vecs = X; 
% prop_resp = ones(N, 1)/N;

[~, G] = get_reachability(A_in);
n_in = size(A_in, 1); % number of possible alpha's, fixed
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix

%%%% initialize
nu = ones(n_in, 1)/n_in; % initial value for p_alpha. may re-write using theta
theta = zeros(K, 1);

% err_prob = 0.05;
%err_prob = 0.1; % if err_prob is too small, 
%range_gen = (1-2*err_prob)/8;
%g = err_prob + (-range_gen/2+range_gen*rand(J,1)); % initial g
%c = 1-err_prob + (-range_gen/2+range_gen*rand(J,1)); % initial c

c = c_in;
g = g_in;

% J * n_class, prob of positive responses for singleton items and existing attribute profiles
theta_mat = bsxfun(@times, c, ideal_resp) + bsxfun(@times, g, 1-ideal_resp);


err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 1e-2 && itera<1000);


%%%% EM algorithm
while iter_indicator
    
    old_loglik = loglik;

    % N * 1 * n_class, prob for each response pattern in X and each attribute profile
    alpha_rc =  bsxfun(   @times, reshape(nu, [1 1 n_in]), ...
        prod(  bsxfun(@power, reshape(theta_mat, [1 J n_in]), resp_vecs) .* ...
               bsxfun(@power, 1-reshape(theta_mat, [1 J n_in]), 1-resp_vecs), 2  )   );
   
    alpha_rc = bsxfun(@rdivide, alpha_rc, sum(alpha_rc, 3));
    
    % update theta: vector of length K
    for k = 1:K
        tmp_num = 0;
        tmp_denom = 0;
        for i = 1:N
            for n = 1:n_in % index for alpha \in A_in
                tmp_num = tmp_num + alpha_rc(i, 1, n) * A_in(n, k) * prod( bsxfun(@power, A_in(n, :)', G(:, k)) );
                tmp_denom = tmp_denom + alpha_rc(i, 1, n) * prod( bsxfun(@power, A_in(n, :)', G(:, k)) );
            end
        end
        theta(k) = tmp_num / tmp_denom;
    end

    % update proportians nu: vector of length n_in
    nu = nonzeros(generate_prop_CBN(binary(0:(2^K-1), K), get_reachability(A_in), theta)); % proportion based on theta


    % N * J, marginal prob for each response pattern 
    prob_know_ri = sum( bsxfun(@times, alpha_rc, ...
        reshape(ideal_resp, [1 J n_in])), 3 );
    
    % update c
    c_denom = sum(prob_know_ri, 1)/N;
    c_nume = sum(resp_vecs .* prob_know_ri, 1)/N;
    
    c(c_denom ~= 0) = c_nume(c_denom ~= 0)' ./ c_denom(c_denom ~= 0)';
    c(c_denom == 0) = 1;
    
    % update g
    g_denom = sum(1-prob_know_ri, 1)/N;
    g_nume = sum(resp_vecs .* (1-prob_know_ri), 1)/N;
    
    g(g_denom ~= 0) = g_nume(g_denom ~= 0)' ./ g_denom(g_denom ~= 0)';
    g(g_denom == 0) = 0;

    % update theta_mat using new c, g
    theta_mat = bsxfun(@times, c, ideal_resp) + bsxfun(@times, g, 1-ideal_resp);
    
    % calculate new loglik
    % the d3_arr has size [N, n_in, J]
    d3_arr = bsxfun(@power, reshape(theta_mat',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta_mat',[1 n_in J]), 1-reshape(X, [N 1 J]));
    % take the product along the third dimension to obtain prod_d2_arr
    prod_d2_arr = prod(d3_arr, 3);
   
    % loglik = sum(ones(N,1) .* log( prod_d2_arr * squeeze(nu) ));
    loglik = sum(log( prod_d2_arr * squeeze(nu)) );
    
    err = (loglik - old_loglik);
        
    iter_indicator = (abs(err) > 2*1e-2 && itera<1000); % tolerance

    itera = itera + 1;
    % fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

n_iter = itera;