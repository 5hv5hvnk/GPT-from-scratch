using Random, Distributions

function model(vocab_size, batch_size, block_size, xb, yb, xb_val, yb_val, max_new_tokens)
    """
    model(vocab_size, batch_size, block_size, xb, yb, xb_val, yb_val, max_new_tokens)

    Train a bigram model using a simple token embedding approach.

    # Arguments
    - `vocab_size`: The size of the vocabulary.
    - `batch_size`: The number of training examples in each batch.
    - `block_size`: The number of tokens in each example.
    - `xb`: The input token sequence for training, represented as a 2D array of shape (batch_size, block_size).
    - `yb`: The target token sequence for training, represented as a 2D array of shape (batch_size, block_size).
    - `xb_val`: The input token sequence for validation, represented as a 2D array of shape (batch_size, block_size).
    - `yb_val`: The target token sequence for validation, represented as a 2D array of shape (batch_size, block_size).
    - `max_new_tokens`: The maximum number of new tokens to generate.

    # Returns
    - `loss`: The average training loss.
    - `loss_validation`: The average validation loss.
    - `new_tokens`: An array of newly generated tokens.

    """
    token_embedding = randn(Float64, vocab_size, vocab_size)
    loss = 0
    loss_validation = 0
    for b in 1:batch_size
        for t in 1:block_size
            probs = exp.(token_embedding[xb[b, t], :]) ./ sum(exp.(token_embedding[xb[b, t], :]))
            loss += logpdf(Categorical(probs), yb[b, t])
            probs_val = exp.(token_embedding[xb_val[b, t], :]) ./ sum(exp.(token_embedding[xb_val[b, t], :]))
            loss_validation += logpdf(Categorical(probs_val), yb_val[b, t])
        end
    end
    loss /= batch_size * block_size
    loss_validation /= batch_size * block_size
    println("************************************************************")
    println("train loss ", -loss, ", val loss ", -loss_validation)
    println("************************************************************")
    new_tokens = Vector{Int}(undef, max_new_tokens)
    new_tokens[1] = 1
    for n in 2:max_new_tokens
        probs_new = exp.(token_embedding[new_tokens[n-1], :]) ./ sum(exp.(token_embedding[new_tokens[n-1], :]))
        new_tokens[n] = rand(Categorical(probs_new))
    end
    return loss, loss_validation, new_tokens
end

vocab_size = 65
batch_size = 32
block_size = 8
max_new_tokens = 10

Random.seed!(314) #pi :))

xb = rand(1:vocab_size, (batch_size, block_size))
yb = rand(1:vocab_size, (batch_size, block_size))
xb_val = rand(1:vocab_size, (batch_size, block_size))
yb_val = rand(1:vocab_size, (batch_size, block_size))

loss, loss_validation, new_tokens = model(vocab_size, batch_size, block_size, xb, yb, xb_val, yb_val, max_new_tokens)
