function [ind_true] = get_ind_from_hier(adj_mat)

% This function generates the true attribute patterns from an K*K adjacency
% matrix which specifies the prerequisite relations between attributes

K = size(adj_mat,1);

A_all = binary(0:(2^K-1), K);

ind_impos = [];
for i=1:K
    for j=(i+1):K
        if adj_mat(i,j)==1
            ind_impos_temp = find(A_all(:,i) < A_all(:,j));
            ind_impos = union(ind_impos, ind_impos_temp);
        end
        
        if adj_mat(j,i)==1
            ind_impos_temp = find(A_all(:,j) < A_all(:,i));
            ind_impos = union(ind_impos, ind_impos_temp);
        end
    end
end

ind_true = setdiff(1:(2^K), ind_impos);
end