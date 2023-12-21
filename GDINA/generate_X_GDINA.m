function [X, X_A] = generate_X_GDINA(N, nu_true, Q, delta_true, A)

[J, K] = size(Q);
% n_in = size(nu_true, 1);
n_in = max(size(nu_true));

% generate multinomial counts
counts = mnrnd(N, nu_true);
X_A = zeros(N, K);
n = 1;
% X_A stores empirical CDF for categories: 1,2,3,...,2^K
for a = 1:length(nu_true)
    X_A(n:(n+counts(a)-1), 1:K) = repmat(A(a, 1:K), counts(a), 1);
    n = n+counts(a);
end
% permute A, size N by 1
X_A = X_A(randperm(N), 1:K);

expand_XA = prod(bsxfun(@power, reshape(X_A, [N 1 K]), reshape(A, [1 n_in K])), 3); % N * n_in
% expand_XA = [ones(N, 1), expand_XA]; % M matrix in the GDINA paper


%%%%%%%%%% generate GDINA parameters %%%%%%%%%%%
p_correct = expand_XA * delta_true';

X = double(rand(size(p_correct)) < p_correct);

end