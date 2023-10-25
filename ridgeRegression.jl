using LinearAlgebra

# Function to calculate solution of ridge regression model Ax = b with 
# ridge parameter lambda
function ridge_regression_solve(A,b;lambda=1e-6)
    (N,d) = size(A)
    if d <= N
        x = (A'*A + lambda*I)\(A'*b)
    else
        x = A'*((A*A' + lambda*I)\b)
    end
    return x
end

# Function to calculate the generalized cross validation (GCV) for ridge regression
function gcv_ridge_regression(A,b;lambda=1e-6)
    x_ridge = ridge_regression_solve(A,b,lambda=lambda)
    GCV = (norm(b - A*x_ridge)^2)/(tr(I - A*((A'*A + lambda*I)\(A')))^2)
    return GCV
end
