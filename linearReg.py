import torch

def createDataset():
    X = torch.Tensor([1,2,3,4,5,6,7,8,9,10])
    y = X * 100  # True weight is 10
    y = y.reshape(-1, 1)
    X = X.reshape(-1, 1)
    return X, y

def forward(weights, bias, X):
    return torch.matmul(X, weights) + bias

def loss_fn(Y_pred, Y):
    loss = 0.5 * (Y_pred - Y)**2
    return torch.mean(loss)

def init_w(X):
    # Initialize with requires_grad=True
    weights = torch.ones((X.shape[1], 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    return weights, bias

def optimizer_step(weights, bias, lr=0.01):  # Increased learning rate
    with torch.no_grad():
        weights -= lr * weights.grad
        bias -= lr * bias.grad

# Training


def main():

    X, y = createDataset()
    weights, bias = init_w(X)
    
    for i in range(1000):
        Y_pred = forward(weights, bias, X)
        loss = loss_fn(Y_pred, y)
        
        if i % 100 == 0:  # Print less frequently
            print(f"Iteration {i}, Loss: {loss.item():.4f}, Weight: {weights.detach().item():.4f}, Bias: {bias.detach().item():.4f}")
        
        # Zero gradients before backward pass
        if weights.grad is not None:
            weights.grad.zero_()
        if bias.grad is not None:
            bias.grad.zero_()
            
        loss.backward()
        optimizer_step(weights, bias)

if __name__=='__main__':
    main()