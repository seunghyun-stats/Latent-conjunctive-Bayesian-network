function [nu, theta, delta, loglik] = get_GDINA_PEM(X, Q, A_in, nu_in, lambda, theta_in, thres_c)

% This function is the GDINA specification of Algorithm 1 in the main paper
%
% @param X: binary item response (N x J)
% @param Q: true Q-matrix (J x K)
% @param A_in: matrix whose rows collect permissible patterns (n_in x K; n_in = 2^K for default use)
% @param nu_in: initial value of the proportion parameters (n_in x 1)
% @param lambda: penalty weight (negative scalar)
% @param theta_in: initial value of DINA item parameter g (J x 1)
% @param thres_c: small positive constant, set to thres_c = 0.01 in simulations

%%%% definitions
[J, K] = size(Q);

N = size(X, 1);
resp_vecs = X; 
n_in = size(A_in, 1); % number of possible alpha's, fixed
A = A_in;

%%%% initialize
% nu = ones(n_in, 1)/n_in; % initial value for p_alpha

nu = nu_in;
theta = theta_in;
delta = zeros(J, n_in);

expand_Q = zeros(J, n_in);
for j = 1:J
    k_index = nonzeros((1:K) .* (Q(j, :) == 1))';
    sub_attributes = unique(A(:, k_index), 'rows');

    for jj = 1:length(sub_attributes)
        n_index = find(ismember(A(:, k_index), sub_attributes(jj, :), 'rows'), 1);
        expand_Q(j, n_index) = 1;
    end
end

attr_combo = A(2:n_in, 1:K);
expand_A = [ones(size(A,1), 1), prod(bsxfun(@power, reshape(A, [size(A,1) 1 K]), ...
    reshape(attr_combo, [1 n_in - 1 K])), 3)]; % n_in * n_in. same as rechability matrix?

err = 1;
itera = 0;

d3_arr = bsxfun(@power, reshape(theta',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta',[1 n_in J]), 1-reshape(X, [N 1 J]));
loglik = sum(log( prod(d3_arr, 3) * squeeze(nu) ));

iter_indicator = (abs(err) > 5*1e-2 && itera<1000);


%%%% EM algorithm
while iter_indicator
    
    old_loglik = loglik;

    % N * 1 * n_class, prob for each response pattern in X and each attribute profile
    alpha_rc =  bsxfun(   @times, reshape(nu, [1 1 n_in]), ...
        prod(  bsxfun(@power, reshape(theta, [1 J n_in]), resp_vecs) .* ...
               bsxfun(@power, 1-reshape(theta, [1 J n_in]), 1-resp_vecs), 2  )   );
   
    alpha_rc = bsxfun(@rdivide, alpha_rc, sum(alpha_rc, 3));


    % update nu: vector of length 2^K
    for n = 1:n_in
        nu(n) = max(thres_c, lambda + ones(N,1)' * alpha_rc(:, n));
    end
    nu = nu/sum(nu);
        
    % update GDINA parameters: theta
    % delta is updated after the while loop
    S = zeros(J, 1);
    for j = 1:J
        k_index = nonzeros((1:n_in) .* expand_Q(j,:));
        unique_rows = unique(expand_A(:, k_index), 'rows', 'stable');  % without stable, order changes...
        S(j) = length(unique_rows); % K_j, number of effective DINA parameters for item j
        % this part of code can be moved outside the while loop

        Rj = zeros(S(j), 1);
        Ij = zeros(S(j), 1);

        for jj = 1:S(j)
            ind_jj = find(ismember(expand_A(:, k_index), unique_rows(jj, :), 'rows'));
            
            Rj(jj) = sum(alpha_rc(:, ind_jj)' * X(:, j)); % this is 0 sometimes..
            Ij(jj) = max(sum(alpha_rc(:, ind_jj)' * ones(N, 1)), 0.1);
            theta(j, ind_jj) = ones(length(ind_jj), 1)' * Rj(jj) / Ij(jj);
        end
    end

    
    % calculate new loglik
    d3_arr = bsxfun(@power, reshape(theta',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta',[1 n_in J]), 1-reshape(X, [N 1 J]));
    prod_d2_arr = prod(d3_arr, 3);
   
    % loglik = sum(ones(N,1) .* log( prod_d2_arr * squeeze(nu) ));
    loglik = sum(log( prod_d2_arr * squeeze(nu)) ) + lambda*sum(log(nu));
    
    err = (loglik - old_loglik);
        
    iter_indicator = (abs(err) > 5*1e-2 && itera<1000); % tolerance

    itera = itera + 1;
    % fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
    % fprintf('PEM iter %d,\t size above threshold %d,\t Err %1.8f\n', itera, sum(nu>0.5/N), err)
end

% actual loglik
loglik = sum(log( prod_d2_arr * squeeze(nu)));

% update delta
for j = 1:J
    delta(j, : ) = expand_A \ theta(j, :)';
end



