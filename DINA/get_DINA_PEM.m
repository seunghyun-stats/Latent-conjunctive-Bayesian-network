function [nu, c, g, lik, n_iter] = get_DINA_PEM(X, Q, A, lambda, c_ini, g_ini, nu_ini, thres_c)

[J, K] = size(Q);

N = size(X, 1);

resp_vecs = X;
prop_resp = ones(N, 1)/N;

c = c_ini;
g = g_ini;

err = 1;
itera = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%
n_in = size(A, 1);
ideal_resp = prod(bsxfun(@power, reshape(A, [1 n_in K]), reshape(Q, [J 1 K])), 3);
nu = nu_ini;
%%%%%%%%%%%%%%%%%%%%%%%%%%

obj_func = 0;
obj_vec = [];
prod_d2_arr = 0;

while abs(err) > 5*1e-2 && itera < 1000
    
    old_obj_func = obj_func;
    
    theta_mat = bsxfun(@times, c, ideal_resp) + bsxfun(@times, g, 1-ideal_resp); 
    
    % prob for each response pattern in X and each attribute profile, under
    % current values of g and c
    
    alpha_rc =  bsxfun(   @times, reshape(nu, [1 1 n_in]), ...
        prod(  bsxfun(@power, reshape(theta_mat, [1 J n_in]), resp_vecs) .* ...
               bsxfun(@power, 1-reshape(theta_mat, [1 J n_in]), 1-resp_vecs), 2  )   );
    alpha_rc = bsxfun(@rdivide, alpha_rc, sum(alpha_rc, 3));

    % update nu: vector of length 2^K
    for n = 1:n_in
        nu(n) = max(thres_c, lambda + ones(N,1)' * alpha_rc(:, n));
    end
    nu = nu/sum(nu);

    prob_know_ri = sum( bsxfun(@times, alpha_rc, ...
        reshape(ideal_resp, [1 J n_in])), 3 );
    
    % update c, g
    c_denom = sum(prob_know_ri, 1)/N;
    c_nume = sum(resp_vecs .* prob_know_ri, 1)/N;
    
    c(c_denom ~= 0) = c_nume(c_denom ~= 0)' ./ c_denom(c_denom ~= 0)';
    c(c_denom == 0) = 1;
   
    g_denom = sum(1-prob_know_ri, 1)/N;
    g_nume = sum(resp_vecs .* (1-prob_know_ri), 1)/N;
    
    g(g_denom ~= 0) = g_nume(g_denom ~= 0)' ./ g_denom(g_denom ~= 0)';
    g(g_denom == 0) = 0;
    itera = itera + 1;
    
    % compute the log likelihood at the current iteration
    d3_arr = bsxfun(@power, reshape(theta_mat',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta_mat',[1 n_in J]), 1-reshape(X, [N 1 J]));
   
    prod_d2_arr = prod(d3_arr, 3);
    
    
    obj_func = sum(ones(N,1) .* log( prod_d2_arr * nu )) + lambda*sum(log(nu));
    err = (obj_func - old_obj_func);
    
    obj_vec = [obj_vec, obj_func];
    
    % fprintf('PEM iter %d,\t size above threshold %d,\t Err %1.8f\n', itera, sum(nu>0.5/N), err);
end

lik = sum(ones(N,1) .* log( prod_d2_arr * nu ));
n_iter = itera;

end