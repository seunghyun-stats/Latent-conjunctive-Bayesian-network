function [t, nu, theta, delta, loglik] = get_CBN_GDINA_EM(X, Q, A_in, G)
%
% This function corresponds to Algorithm 2 in the main paper.

%%%% definitions
[J, K] = size(Q);

N = size(X, 1);
resp_vecs = X; 
% [~, G] = get_reachability(A_in);
n_in = size(A_in, 1); % number of possible alpha's, fixed
A = A_in;

%%%% initialize
nu = ones(n_in, 1)/n_in; % initial value for p_alpha. may re-write using theta
t = zeros(K, 1);
g = 0.1;
c = 0.9;
% delta = [ones(J, 1) * g, ones(J, n_in - 1)/(n_in - 1)*(c - g)];
theta = [ones(J, 1) * g, ones(J, n_in - 1) * g + repmat(1:(n_in - 1),J,1)/(n_in - 1)*(c - g)];
delta = zeros(J, n_in);

% J * n_class, prob of positive responses for singleton items and existing attribute profiles
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

% expand_Q_all = [ones(M,1), prod(bsxfun(@power, reshape(Q, [J 1 K]), ...
%     reshape(get_I(K, K), [1 2^K-1 K])), 3)];
% ok_ex_all = (expand_Q==1); % J * 2^K

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


    % update t: vector of length K
    for k = 1:K
        tmp_num = 0;
        tmp_denom = 0;
        for i = 1:N
            for n = 1:n_in % index for alpha \in A_in
                tmp_num = tmp_num + alpha_rc(i, 1, n) * A_in(n, k) * prod( bsxfun(@power, A_in(n, :)', G(:, k)) );
                tmp_denom = tmp_denom + alpha_rc(i, 1, n) * prod( bsxfun(@power, A_in(n, :)', G(:, k)) );
            end
        end
        t(k) = tmp_num / tmp_denom;
    end

    % update proportians nu: vector of length n_in
    nu = nonzeros(generate_prop_CBN(binary(0:(2^K-1), K), G, t)); % proportion based on theta

        
    % update GDINA parameters: theta
    % delta is updated after the while loop
    S = zeros(J, 1);
    for j = 1:J
        k_index = nonzeros((1:n_in) .* expand_Q(j,:));
        unique_rows = unique(expand_A(:, k_index), 'rows', 'stable');
        S(j) = length(unique_rows); % K_j, number of effective DINA parameters for item j
        % this part of code can be moved outside the while loop

        Rj = zeros(n_in, 1);
        Ij = zeros(n_in, 1);

        for jj = 1:S(j)
            ind_jj = find(ismember(expand_A(:, k_index), unique_rows(jj, :), 'rows'));
            
            Rj(jj) = sum(alpha_rc(:, ind_jj)' * X(:, j));
            Ij(jj) = sum(alpha_rc(:, ind_jj)' * ones(N, 1));
            theta(j, ind_jj) = ones(length(ind_jj), 1)' * Rj(jj) / Ij(jj);
        end
    end

    
    % calculate new loglik
    d3_arr = bsxfun(@power, reshape(theta',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta',[1 n_in J]), 1-reshape(X, [N 1 J]));
    prod_d2_arr = prod(d3_arr, 3);
   
    % loglik = sum(ones(N,1) .* log( prod_d2_arr * squeeze(nu) ));
    loglik = sum(log( prod_d2_arr * squeeze(nu)) );
    
    err = (loglik - old_loglik);
        
    iter_indicator = (abs(err) > 4*1e-2 && itera<1000); % tolerance

    itera = itera + 1;
    fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end

% update delta
for j = 1:J
    delta(j, : ) = expand_A \ theta(j, :)';
end



