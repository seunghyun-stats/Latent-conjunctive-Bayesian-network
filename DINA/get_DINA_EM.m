function [nu, c, g, loglik] = get_DINA_EM(X, Q, A_in, err_prob)

% EM algorithm for estimating c, g and mixing coeffcients \pi

[J, K] = size(Q);


N = size(X, 1);
resp_vecs = X; 
prop_resp = ones(N, 1)/N;

%%%%%%%%%%%%%%%%%%%%%%%%%%
n_in = size(A_in, 1);
ideal_resp = prod(bsxfun(@power, reshape(A_in, [1 n_in K]), ...
    reshape(Q, [J 1 K])), 3); % ideal response matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%

nu = ones(n_in, 1)/n_in;

% err_prob = r;
range_gen = (1-2*err_prob)/8;
g = err_prob + (-range_gen/2+range_gen*rand(J,1));
c = 1-err_prob + (-range_gen/2+range_gen*rand(J,1));


err = 1;
itera = 0;
loglik = 0;

iter_indicator = (abs(err) > 5*1e-2 && itera<1000);

while iter_indicator
    
    old_loglik = loglik;
    
    % J * n_class, prob of positive responses for singleton items and existing attribute profiles
    theta_mat = bsxfun(@times, c, ideal_resp) + bsxfun(@times, g, 1-ideal_resp);

    alpha_rc =  bsxfun(   @times, reshape(nu, [1 1 n_in]), ...
        prod(  bsxfun(@power, reshape(theta_mat, [1 J n_in]), resp_vecs) .* ...
               bsxfun(@power, 1-reshape(theta_mat, [1 J n_in]), 1-resp_vecs), 2  )   );
    
    % N * 1 * (n_class+1), prob for each response pattern in X and each attribute profile
    alpha_rc = bsxfun(@rdivide, alpha_rc, sum(alpha_rc, 3));
    
    nu = sum(bsxfun(@times, alpha_rc, prop_resp), 1);
    
    
    % N * J, marginal prob for each response pattern 
    prob_know_ri = sum( bsxfun(@times, alpha_rc, ...
        reshape(ideal_resp, [1 J n_in])), 3 );
    
    
    c_denom = sum(prob_know_ri, 1)/N;
    c_nume = sum(resp_vecs .* prob_know_ri, 1)/N;
    
    c(c_denom ~= 0) = c_nume(c_denom ~= 0)' ./ c_denom(c_denom ~= 0)';
    c(c_denom == 0) = 1;
    
    g_denom = sum(1-prob_know_ri, 1)/N;
    g_nume = sum(resp_vecs .* (1-prob_know_ri), 1)/N;
    
    g(g_denom ~= 0) = g_nume(g_denom ~= 0)' ./ g_denom(g_denom ~= 0)';
    g(g_denom == 0) = 0;
    
%     err0 = abs(old_c-c)+abs(old_g-g);
% 
%     ind_item = (g_denom~=0 .* c_denom~=0);
% 
%     err = max(err0(ind_item));     
%     if isempty(err)
%         break
%     end
    
    % the d3_arr has size [N, n_in, J]
    d3_arr = bsxfun(@power, reshape(theta_mat',[1 n_in J]), reshape(X, [N 1 J])) .* ...
        bsxfun(@power, 1-reshape(theta_mat',[1 n_in J]), 1-reshape(X, [N 1 J]));
    % take the product along the third dimension to obtain prod_d2_arr
    prod_d2_arr = prod(d3_arr, 3);
    
    
    loglik = sum(ones(N,1) .* log( prod_d2_arr * squeeze(nu) ));
    
    
    err = (loglik - old_loglik);
    
        
    iter_indicator = (abs(err) > 5* 1e-2 && itera<1000);
    
    
    itera = itera + 1;
   %fprintf('EM Iteration %d,\t Err %1.8f\n', itera, err);
end


nu = nu/sum(nu);

nu = squeeze(nu);



end