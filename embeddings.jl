using LinearAlgebra, Distributions
lm_head(x, A, b) = A * x .+ b 

function softmax(x)
    e_x = exp.(x .- maximum(x))
    return e_x ./ sum(e_x)
end

vocab_size = 65
batch_size = 32
block_size = 8
n_embed = 32
xb = rand(1:vocab_size, batch_size, block_size)
yb = rand(1:vocab_size, batch_size, block_size)
xb_val = rand(1:vocab_size, batch_size, block_size)
yb_val = rand(1:vocab_size, batch_size, block_size)
max_new_tokens = 100

token_embedding = rand(n_embed, vocab_size)
lm_head_multiplier = rand(vocab_size, n_embed)
lm_head_offset = rand(vocab_size)
function calculate_loss(xb, yb, batch_size, block_size, vocab_size, 
                        token_embedding, lm_head_multiplier, lm_head_offset)
    loss = 0.0
    for b in 1:batch_size
        for t in 1:block_size
            logits = lm_head(token_embedding[:, xb[b, t]], lm_head_multiplier, lm_head_offset)
            dist = Categorical(softmax(logits))
            loss -= logpdf(dist, yb[b, t])
        end
    end
    return loss / (batch_size * block_size)
end

loss = calculate_loss(xb, yb, batch_size, block_size, vocab_size, 
                      token_embedding, lm_head_multiplier, lm_head_offset)
loss_validation = calculate_loss(xb_val, yb_val, batch_size, block_size, vocab_size, 
token_embedding, lm_head_multiplier, lm_head_offset)

print("************************************************************\n");
print("train loss ", -loss, ", val loss ", -loss_validation,"\n");
print("************************************************************\n");

function generate_new_tokens(token_embedding, lm_head_multiplier, lm_head_offset, max_new_tokens, vocab_size)
    new_tokens = fill(1, max_new_tokens)
    for n in 2:max_new_tokens
        logits = lm_head(token_embedding[:, new_tokens[n - 1]], lm_head_multiplier, lm_head_offset)
        dist = Categorical(softmax(logits))
        new_tokens[n] = rand(dist)
    end
    return new_tokens
end

new_tokens = generate_new_tokens(token_embedding, lm_head_multiplier, lm_head_offset, max_new_tokens, vocab_size)
