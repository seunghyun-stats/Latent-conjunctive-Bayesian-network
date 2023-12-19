function [prop_vec] = generate_prop_CBN(A_all, reach_mat, theta_posi)


K = size(A_all, 2);
prop_vec = zeros(2^K, 1);

for aa = 1:2^K
    [is_allowed, next_attr] = get_next_attr(A_all(aa,:), reach_mat);
    if is_allowed == 0
        prop_vec(aa) = 0;
    else
        prop_vec(aa) = prod(theta_posi(A_all(aa,:)==1)) * prod(1 - theta_posi(next_attr));
    end
end

end