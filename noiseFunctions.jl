using LinearAlgebra
using Random

# Function returning a function representing scalar zero noise
function zero_noise()
    noise(x) = return 0
    return noise
end

# Function returning a function representing scalar Gaussian random noise
function gaussian_random_noise(;mu=0,sigma=1)
    noise(x) = mu + sigma*randn()
    return noise
end
