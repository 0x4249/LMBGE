using LinearAlgebra
include("ridgeRegression.jl")

# Function to estimate gradient using standard central finite differences 
function cfd_grad(func_eval,x;h=1e-3)
    d = length(x)
    V_pos = Matrix{Float64}(I,d,d) 
    f_pos = zeros(d)
    f_neg = zeros(d)
    for j in 1:d
        f_pos[j] = func_eval(x + h*V_pos[:,j])
        f_neg[j] = func_eval(x - h*V_pos[:,j])
    end
    cfd_grad = (f_pos - f_neg)/(2*h)
    func_eval_count = 2*d
    return cfd_grad, func_eval_count
end

# Function to estimate gradient using linear model based gradient estimation (LMBGE) with stencil gradient
function lmbge_stencil_grad(func_eval,x,V;h=1e-3,beta=1e-5,g_prev=nothing)
    d = length(x)
    f_0 = func_eval(x)
    N = size(V,2)
    f_vs = zeros(N)
    for j in 1:N
        f_vs[j] = func_eval(x + h*V[:,j])
    end
    if isnothing(g_prev)
        g_prev = zeros(d)
    end
    df_h = (f_vs .- f_0)/h
    df_h_shifted = df_h - V'*g_prev
    stencil_grad = g_prev + ridge_regression_solve(V', df_h_shifted, lambda=beta)
    residual = V'*stencil_grad - df_h
    sample_rmse = norm(residual)/sqrt(N)
    func_eval_count = N + 1
    return stencil_grad, func_eval_count, sample_rmse
end

# Function to estimate both function and gradient using linear model based gradient estimation (LMBGE)
function lmbge_func_and_grad(func_eval,x,V;h=1e-3,mu=1e-5,theta_prev=nothing)
    d = length(x)
    N = size(V,2)
    f_vs = zeros(N)
    for j in 1:N
        f_vs[j] = func_eval(x + h*V[:,j])
    end
    V_hat = hcat(ones(N), h*V')
    if isnothing(theta_prev)
        theta_prev = zeros(d+1)
    end
    f_shifted = f_vs - V_hat*theta_prev
    theta = theta_prev + ridge_regression_solve(V_hat, f_shifted, lambda=mu)
    residual = V_hat*theta - f_vs
    sample_rmse = norm(residual)/sqrt(N)
    func_eval_count = N
    return theta, func_eval_count, sample_rmse
end

# Function to estimate gradient using linear model based gradient estimation (LMBGE) with stencil gradient with regularization parameter chosen via generalized cross validation (GCV)
function gcv_lmbge_stencil_grad(func_eval,x,V;h=1e-3,g_prev=nothing)
    d = length(x)
    f_0 = func_eval(x)
    N = size(V,2)
    f_vs = zeros(N)
    for j in 1:N
        f_vs[j] = func_eval(x + h*V[:,j])
    end
    if isnothing(g_prev)
        g_prev = zeros(d)
    end
    df_h = (f_vs .- f_0)/h
    df_h_shifted = df_h - V'*g_prev
    
    # Optimize GCV via grid search in log space
    lambdas = 10 .^ range(-5,stop=5,length=11)
    num_vals = length(lambdas)
    GCV_vals = zeros(num_vals)
    for i in 1:num_vals
        GCV_vals[i] = gcv_ridge_regression(V', df_h_shifted, lambda=lambdas[i])
    end
    
    index_min = argmin(GCV_vals)
    stencil_grad = g_prev + ridge_regression_solve(V', df_h_shifted, lambda=lambdas[index_min])
    
    residual = V'*stencil_grad - df_h
    sample_rmse = norm(residual)/sqrt(N)
    func_eval_count = N + 1
    return stencil_grad, func_eval_count, sample_rmse
end

# Function to estimate both function and gradient using linear model based gradient estimation (LMBGE) with regularization parameter chosen via generalized cross validation (GCV)
function gcv_lmbge_func_and_grad(func_eval,x,V;h=1e-3,theta_prev=nothing)
    d = length(x)
    N = size(V,2)
    f_vs = zeros(N)
    for j in 1:N
        f_vs[j] = func_eval(x + h*V[:,j])
    end
    V_hat = hcat(ones(N), h*V')
    if isnothing(theta_prev)
        theta_prev = zeros(d+1)
    end
    f_shifted = f_vs - V_hat*theta_prev
    
    # Optimize GCV via grid search in log space
    lambdas = 10 .^ range(-5,stop=5,length=11)
    num_vals = length(lambdas)
    GCV_vals = zeros(num_vals)
    for i in 1:num_vals
        GCV_vals[i] = gcv_ridge_regression(V_hat, f_shifted, lambda=lambdas[i])
    end
    
    index_min = argmin(GCV_vals)
    theta = theta_prev + ridge_regression_solve(V_hat, f_shifted, lambda=lambdas[index_min])
    
    residual = V_hat*theta - f_vs
    sample_rmse = norm(residual)/sqrt(N)
    func_eval_count = N
    return theta, func_eval_count, sample_rmse
end
