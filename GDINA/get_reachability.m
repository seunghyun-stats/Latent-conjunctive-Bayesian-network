function [reach_mat, adj_mat] = get_reachability(patterns)

% patterns = sortrows(patterns')';
[~, K] = size(patterns);

reach_mat = zeros(K, K);


% -- get the dense reachability matrix -- %
for i = 1:K
    for j = i+1:K
        reach_mat(i,j) = all(patterns(:,i) >= patterns(:,j));
    end
end


% -- remove implied indirect prerequisite relations -- %
adj_mat = reach_mat;
for k=2:K
    adj_mat(reach_mat^k ~= 0) = 0;
end



end