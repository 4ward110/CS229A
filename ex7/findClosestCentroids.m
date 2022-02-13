function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% 
% for i=1:size(X,1)
%     g1 = X(i, :);
%     a = [];
%     for j=1:K
%         g2 = centroids(j, :);
%         norm_cal = norm(g2-g1);
%         a(j) = norm_cal;
%     end
%     min_ind = find(a == min(a));
%     idx(i) = min_ind;
% end
% 


for i = 1:size(X,1)
  x = X(i,:);
  max = inf;
  ind = 0;
  for j = 1:K
    cent = centroids(j,:);
    cost = sum((x - cent) .^ 2);
    if cost < max,
      max = cost;
      ind = j;
    end
    idx(i,1) = ind;
  end
end
% =============================================================

end

