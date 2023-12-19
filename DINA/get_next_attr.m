function [is_allowed, next_attr] = get_next_attr(pattern, reach_mat)

% Given an arbitrary binary pattern and an attribute hierarchy, this function
% returns the immediate next attributes that can be present

K = length(pattern);
next_attr = zeros(1,K);

is_allowed = 1;

for k = 1:K
    k_parents = (reach_mat(:,k)==1);
    
    % check if pattern is allowed under the attribute hierarchy
    if pattern(k) == 1
        if any(pattern(k_parents) == 0)
            is_allowed = 0;
        end 
    end
    
    if pattern(k) == 0
        % k_parents can be empty
        if all(pattern(k_parents) == 1)
            next_attr(k) = 1;
        end
    end
end

if is_allowed == 0
    next_attr = zeros(1,K);
end

next_attr = logical(next_attr);

end