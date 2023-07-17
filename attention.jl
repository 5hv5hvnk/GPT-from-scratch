using Distributions, LinearAlgebra

lm_head(x, A, b) = A * x + b

function self_attention(x, key, query, value)
    block_size = size(x, 1)
    n_embed = size(key, 1)
    head_size = size(key, 2)

    x_matrix = hcat(x...)
    k = x_matrix * key
    q = x_matrix * query
    v = x_matrix * value

    wei = zeros(block_size, block_size)
    tmp_wei = q * k' / sqrt(head_size)
    for t in 1:block_size
        wei[t, 1:t] = softmax(tmp_wei[t, 1:t])
    end

    weighted_value = wei * v
    out = [weighted_value[t, :] for t in 1:block_size]
    return out
end

function self_attention(x, key, query, value)
    batch_size = size(x, 1)
    block_size = size(x, 2)
    n_embed = size(key, 1)
    head_size = size(key, 2)

    out = [self_attention(x[b], key, query, value) for b in 1:batch_size]
    return out
end

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
    key::Array{Float64,2}
    query::Array{Float64,2}
    value::Array{Float64,2}
end

function model(data::Data, params::Parameters)
    x = [params.token_embedding[data.xb[b, t]] + params.position_embedding[t] for b in 1:data.batch_size, t in 1:data.block_size]

    x = self_attention(x, params.key, params.query, params.value)

    loss = 0.0
    for b in 1:data.batch_size
        for t in 1:data.block_size
            logits = lm_head(x[b][t], params.lm_head_multiplier, params.lm_head_offset)
            loss += logpdf(Categorical(logits), data.yb[b, t])
        end
    end
    loss /= data.batch_size * data.block_size

    return loss
end

function generated_quantities(data::Data, params::Parameters)
    x_val = [params.token_embedding[data.xb_val[b, t]] + params.position_embedding[t] for b in 1:data.batch_size, t in 1:data.block_size]

    x_val = self_attention(x_val, params.key, params.query, params.value)

    loss_validation = 0.0
    for b in 1:data.batch_size
        for t in 1:data.block_size
            logits = lm_head(x_val[b][t], params.lm_head_multiplier, params.lm_head_offset)
            loss_validation += logpdf(Categorical(logits), data.yb_val[b, t])
        end
    end
    loss_validation /= data.batch_size * data.block_size

    new_tokens = zeros(Int64, data.max_new_tokens)
    new_tokens[1] = 1
    x_new = [zeros(data.n_embed) for _ in 1:data.block_size]
    for n in 2:data.max_new_tokens
        x_new = [zeros(data.n_embed) for _ in 1:data.block_size]
        for t in 1:min(n - 1, data.block_size)
            x_new[t] = params.token_embedding[new_tokens[max(0, n - 1 - data.block_size) + t]] + params.position_embedding[t]
        end

        x_new = self_attention(x_new, params.key, params.query, params.value)

        new_tokens[n] = rand(Categorical(lm_head(x_new[min(n - 1, data.block_size)], params.lm_head_multiplier, params.lm_head_offset)))
    end

    println("************************************************************")
    println("train loss ", -model(data, params), ", val loss ", -loss_validation)
    println("************************************************************")

    return new_tokens
end
