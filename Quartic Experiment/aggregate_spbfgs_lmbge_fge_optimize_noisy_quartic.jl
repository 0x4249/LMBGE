using CUTEst
using LinearAlgebra
using Printf
using PyPlot
using Random
using Statistics
include("../gradientEstimators.jl")
include("../noiseFunctions.jl")
include("../objectiveFunctions.jl")

# Define gradient estimator
h = 1e-3
function grad_estimate(func_eval,x,h;use_gcv=true,theta_prev=nothing)
    d = length(x)
    V_pos = Matrix{Float64}(I,d,d) 
    V = hcat(V_pos, -V_pos)
    if use_gcv
        return gcv_lmbge_func_and_grad(func_eval,x,V,h=h,theta_prev=theta_prev)
    else
        return lmbge_func_and_grad(func_eval,x,V,h=h,mu=1e-5,theta_prev=theta_prev)
    end
end

# Initialize objective function
nlp = CUTEstModel("TQUARTIC", "-param", "N=10")
d = length(nlp.meta.x0)
objFunTrue = nlp_objective(nlp)

# Set minimum objective value according to SIF file for CUTEst problem above
obj_fun_min_val = 0.0

# Set function noise level
sigma_true_f = (1e-4)*abs(objFunTrue.func_eval(nlp.meta.x0))

# Set noise type
objFun = nlp_objective_additive_noise(nlp,gaussian_random_noise(sigma=sigma_true_f),zero_noise())

# Aggregate parameters
total_runs = 30
maxIter = 60
curv_fail_counts = zeros(total_runs)
line_search_fail_counts = zeros(total_runs)
obj_fun_eval_counts = zeros(total_runs)
all_f_vals = zeros(maxIter+1,total_runs)
all_grad_est_norm_vals = zeros(maxIter+1,total_runs)
all_x_vals = zeros(maxIter+1,d,total_runs)
all_rmse_vals = zeros(maxIter+1,total_runs)
all_sigma_est_vals = zeros(maxIter+1,total_runs)
all_scaled_hess_cond_vals = zeros(maxIter,total_runs)
all_line_search_fails = zeros(maxIter,total_runs)
all_alphas = zeros(maxIter,total_runs)
all_betas = zeros(maxIter,total_runs)

# Plotting stuff
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["figure.autolayout"] = true
color_map = get_cmap("twilight_shifted", total_runs)

# Repeated algorithm runs
for r in 1:total_runs
    global all_f_vals, all_x_vals, all_line_search_fails, all_betas, maxIter

    # Optimization setup
    x_0 = nlp.meta.x0
    H_0 = Matrix{Float64}(I,d,d)
    
    # Backtracking line search parameters
    alpha_0 = 1
    c_1 = 1e-4
    tau = 0.1
    max_backtracks = 10
	
    # Allocate storage
    curv_fail_count = 0
    line_search_fail_count = 0
    obj_fun_eval_count = 0
    f_vals = zeros(maxIter+1)
    grad_est_norm_vals = zeros(maxIter+1)
    x_vals = zeros(maxIter+1,d)
    rmse_vals = zeros(maxIter+1)
    sigma_est_vals = zeros(maxIter+1)
    scaled_hess_cond_vals = zeros(maxIter)
    line_search_fails = zeros(maxIter)
    alphas = zeros(maxIter)
    betas = zeros(maxIter)

    # SP-BFGS Loop
    x = x_0
    H = H_0
    f_old = Inf
    g_old = zeros(d)
    theta_old = vcat(f_old, g_old)	
    rmse_old = Inf
    sigma_est_old = Inf

    for i in 1:maxIter
        alpha = alpha_0
        
        # Initial noisy function evaluation and gradient estimation
	if i == 1
	    theta_old, g_func_eval_count, rmse_old = grad_estimate(objFun.func_eval,x,h,use_gcv=false)
	    f_old = theta_old[1]
            g_old = theta_old[2:end]
	    obj_fun_eval_count += g_func_eval_count
	    sigma_est_old = rmse_old
			
	    f_vals[1] = f_old
	    grad_est_norm_vals[1] = norm(g_old)
            x_vals[1,:] = x
	    rmse_vals[1] = rmse_old
	    sigma_est_vals[1] = sigma_est_old
	end

	x_old = x
	p = -H*g_old
	x = x_old + alpha*p
        
        theta_new, g_func_eval_count, rmse_new = grad_estimate(objFun.func_eval,x,h,theta_prev=theta_old)
        f_new = theta_new[1]
        g_new = theta_new[2:end]
	obj_fun_eval_count += g_func_eval_count
	sigma_est_new = rmse_new
        
	# Backtracking line search
	line_search_fail = 0
	while f_new > f_old + c_1*alpha*dot(g_old,p)
	    line_search_fail +=1
	    if line_search_fail > max_backtracks
	        alpha = 0
		x = x_old
		break
	    else
	        alpha = tau*alpha
	    end
	    x = x_old + alpha*p		 
	    theta_new, g_func_eval_count, rmse_new = grad_estimate(objFun.func_eval,x,h,theta_prev=theta_old)
            f_new = theta_new[1]
            g_new = theta_new[2:end]
            obj_fun_eval_count += g_func_eval_count
	    sigma_est_new = rmse_new
	end
	
	line_search_fail_count += line_search_fail
	line_search_fails[i] = line_search_fail
	alphas[i] = alpha
        
	# Calculate secant condition parameters
	s_k = x - x_old
	y_k = g_new - g_old
	y_k_dot_s_k = dot(s_k,y_k)
        
	# Set penalty parameter
        beta_k = h*norm(s_k)/(sqrt(2)*sigma_est_old)
        
	# Check penalized curvature condition
	if y_k_dot_s_k > -1/beta_k && norm(s_k) > 0
	    betas[i] = beta_k

	    # Calculate damping parameters
	    gamma_k = 1/(y_k_dot_s_k+1/beta_k)
	    omega_k = 1/(y_k_dot_s_k+2/beta_k)
	    @show omega_k, gamma_k

	    Hy = H*y_k
	    y_H_y = dot(y_k,Hy)
	    D_1 = gamma_k/omega_k
	    D_2 = (gamma_k - omega_k)*y_H_y

	    H = (I - omega_k*s_k*y_k')*H*(I - omega_k*y_k*s_k') + omega_k*(D_1 + D_2)*s_k*s_k'
	else 
	    @printf("Curvature condition failed!\n")
	    curv_fail_count += 1
	end

	@show(line_search_fail)
	@show(i,obj_fun_eval_count,f_old,f_new)

	f_vals[i+1] = f_new
	grad_est_norm_vals[i+1] = norm(g_new)
	x_vals[i+1,:] = x
	rmse_vals[i+1] = rmse_new
	sigma_est_vals[i+1] = sigma_est_new
	scaled_hess_cond_vals[i] = cond(H*objFun.hess_eval(x))
	f_old = f_new
	g_old = g_new
	theta_old = theta_new
	rmse_old = rmse_new
	sigma_est_old = sigma_est_new
	@show r
    end

    @show(curv_fail_count)
    @show(line_search_fail_count)
    @show(norm(H))
    @show(norm(objFun.hess_eval(x)))

    curv_fail_counts[r] = curv_fail_count
    line_search_fail_counts[r] = line_search_fail_count
    obj_fun_eval_counts[r] = obj_fun_eval_count
    all_f_vals[:,r] = f_vals
    all_grad_est_norm_vals[:,r] = grad_est_norm_vals
    all_x_vals[:,:,r] = x_vals
    all_rmse_vals[:,r] = rmse_vals
    all_sigma_est_vals[:,r] = sigma_est_vals
    all_scaled_hess_cond_vals[:,r] = scaled_hess_cond_vals
    all_line_search_fails[:,r] = line_search_fails
    all_alphas[:,r] = alphas
    all_betas[:,r] = betas
end

# Make subplots
subplots_fig, axes_array = plt.subplots(2,2)
subplots_fig.suptitle("LMBGE Joint Function And Gradient Estimation - SP-BFGS", fontsize=14)
axes_opt_gap = axes_array[1,1]
axes_sigma = axes_array[1,2]
axes_grad_norm = axes_array[2,1]
axes_grad_est_norm = axes_array[2,2]

# Optimality gap figure
pltIter = maxIter
axes_opt_gap.set_xlim([0,pltIter])
axes_opt_gap.set_ylim([-4,0])
axes_opt_gap.grid(true)
axes_opt_gap.tick_params(axis="x", labelsize=14)
axes_opt_gap.tick_params(axis="y", labelsize=14)
axes_opt_gap.set_xlabel("Iteration k", fontsize=16)
axes_opt_gap.set_ylabel(L"$\log_{10}(\phi_k - \phi^{\star})$", fontsize=16)
axes_opt_gap.set_title("Optimality Gap", fontsize=16)
all_true_obj_fun_vals = zeros(maxIter+1,total_runs)
for r in 1:total_runs
    for j in 1:(maxIter+1)
        all_true_obj_fun_vals[j,r] = objFunTrue.func_eval(all_x_vals[j,:,r])
    end
    axes_opt_gap.plot(0:pltIter,log.(10,all_true_obj_fun_vals[:,r]), alpha=0.8, color=color_map(r/total_runs), label="SP-BFGS")
end

# Sigma estimate figure
axes_sigma.set_xlim([0,pltIter])
axes_sigma.set_ylim([-5.5,-2.5])
axes_sigma.grid(true)
axes_sigma.tick_params(axis="x", labelsize=14)
axes_sigma.tick_params(axis="y", labelsize=14)
axes_sigma.set_xlabel("Iteration k", fontsize=16)
axes_sigma.set_ylabel(L"$\log_{10} \bigg ( \frac{1}{\sqrt{J}} \left\| \mathbf{r}(\widehat{\theta}_k) \right\|_2 \bigg )$", fontsize=16)
axes_sigma.set_title(L"True $\sigma$ Vs. Estimates $\hat{\sigma}_{k}$", fontsize=16)
for r in 1:total_runs
    axes_sigma.plot(0:pltIter,log.(10,all_sigma_est_vals[:,r]), alpha=0.8, color=color_map(r/total_runs), label="SP-BFGS")
end
axes_sigma.axhline(y=log(10,sigma_true_f), color="r", linestyle="dotted", alpha=1.0)

# Gradient norm figure
axes_grad_norm.set_xlim([0,pltIter])
axes_grad_norm.set_ylim([-2.5,2.5])
axes_grad_norm.grid(true)
axes_grad_norm.tick_params(axis="x", labelsize=14)
axes_grad_norm.tick_params(axis="y", labelsize=14)
axes_grad_norm.set_xlabel("Iteration k", fontsize=16)
axes_grad_norm.set_ylabel(L"$\log_{10}(\left\| \nabla \phi_k \right\|_2)$", fontsize=16)
axes_grad_norm.set_title("True Gradient Norm", fontsize=16)
all_grad_norms = zeros(maxIter+1,total_runs)
for r in 1:total_runs
    for j in 1:(maxIter+1)
        all_grad_norms[j,r] = norm(objFunTrue.grad_eval(all_x_vals[j,:,r]))
    end
    axes_grad_norm.plot(0:pltIter,log.(10,all_grad_norms[:,r]), alpha=0.8, color=color_map(r/total_runs), label="SP-BFGS")
end

# Gradient estimate norm figure
axes_grad_est_norm.set_xlim([0,pltIter])
axes_grad_est_norm.set_ylim([-2.5,2.5])
axes_grad_est_norm.grid(true)
axes_grad_est_norm.tick_params(axis="x", labelsize=14)
axes_grad_est_norm.tick_params(axis="y", labelsize=14)
axes_grad_est_norm.set_xlabel("Iteration k", fontsize=16)
axes_grad_est_norm.set_ylabel(L"$\log_{10}(\left\| \mathbf{a}_k \right\|_2)$", fontsize=16)
axes_grad_est_norm.set_title("Gradient Estimate Norm", fontsize=16)
for r in 1:total_runs
    axes_grad_est_norm.plot(0:pltIter,log.(10,all_grad_est_norm_vals[:,r]), alpha=0.8, color=color_map(r/total_runs), label="SP-BFGS")
end

# Print table statistics
@printf("====Table Values====\n")
@printf("SP-BFGS F&GE Mean Number Of Objective Evaluations: %d\n", mean(obj_fun_eval_counts))
@printf("SP-BFGS F&GE Median Number Of Objective Evaluations: %d\n", median(obj_fun_eval_counts))
@printf("SP-BFGS F&GE Mean Log10 Optimality Gap: %1.1E\n", mean(log.(10,all_true_obj_fun_vals[end,:].-obj_fun_min_val)))
@printf("SP-BFGS F&GE Median Log10 Optimality Gap: %1.1E\n", median(log.(10,all_true_obj_fun_vals[end,:].-obj_fun_min_val)))
