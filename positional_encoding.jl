using Distributions

lm_head(x, A, b) = A * x .+ b 

struct Data
    vocab_size::Int64
    batch_size::Int64
    block_size::Int64
    n_embed::Int64
    xb::Array{Int64,2}
    yb::Array{Int64,2}
    xb_val::Array{Int64,2}
    yb_val::Array{Int64,2}
    max_new_tokens::Int64
end

struct Parameters
    token_embedding::Array{Array{Float64,1},1}
    lm_head_multiplier::Array{Float64,2}
    lm_head_offset::Array{Float64,1}
    position_embedding::Array{Array{Float64,1},1}
end

function transformed_parameters_and_loss(data::Data, params::Parameters)
    x = [params.token_embedding[data.xb[b, t]] + params.position_embedding[t] for b in 1:data.batch_size, t in 1:data.block_size]
    
    loss = 0.0
    for b in 1:data.batch_size
        for t in 1:data.block_size
            logits = lm_head(x[b, t], params.lm_head_multiplier, params.lm_head_offset)
            loss += logpdf(Categorical(logits), data.yb[b, t])
        end
    end
    loss /= data.batch_size * data.block_size
    
    return x, loss
end

function model(data::Data, params::Parameters)
    x, loss = transformed_parameters_and_loss(data, params)
    loss
end

function generated_quantities(data::Data, params::Parameters)
    x_val = [params.token_embedding[data.xb_val[b, t]] + params.position_embedding[t] for b in 1:data.batch_size, t in 1:data.block_size]
    
    loss_validation = 0.0
    for b in 1:data.batch_size
        for t in 1:data.block_size
            logits = lm_head(x_val[b, t], params.lm_head_multiplier, params.lm_head_offset)
            loss_validation += logpdf(Categorical(logits), data.yb_val[b, t])
        end
    end
    loss_validation /= data.batch_size * data.block_size
    
    new_tokens = Array{Int64}(undef, data.max_new_tokens)
    new_tokens[1] = 1
    for n in 2:data.max_new_tokens
        x_new = params.token_embedding[new_tokens[n - 1]] + params.position_embedding[min(n - 1, data.block_size)]
        logits = lm_head(x_new, params.lm_head_multiplier, params.lm_head_offset)
        new_tokens[n] = rand(Categorical(logits))
    end
    
    println("************************************************************")
    println("train loss ", -loss_validation, ", val loss ", -loss_validation)
    println("************************************************************")
    
    return new_tokens
end
