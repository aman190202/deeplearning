import mlx.core as mx

def createDataset():
    X = mx.array([1,2,3,4,5,6,7,8,9,10])
    y = X * 10  
    y = y.reshape(-1, 1)
    X = X.reshape(-1, 1)
    return X, y

def forward(weights, bias, X):
    return X @ weights + bias

def loss_fn(weights, bias, X, Y):
    Y_pred = X @ weights + bias
    loss = 0.5 * (Y_pred - Y)**2
    return mx.mean(loss)

def init_w(X):
    # Initialize with requires_grad=True
    weights = mx.random.normal((X.shape[1], 1))
    bias = mx.random.normal((1,))
    return weights, bias


def main():

    X, y = createDataset()
    weights, bias = init_w(X)

    lr = 0.01
    for i in range(1000):

        loss = loss_fn(weights, bias, X, y)
        print(f'loss : {loss}')

        grad_w = mx.grad(loss_fn)
        g_w = grad_w(weights, bias, X, y)


        grad_b = mx.grad(loss_fn,argnums=1)
        g_b = grad_b(weights, bias, X, y)

        weights -= lr * g_w
        bias -= lr * g_b

        print(weights,bias)

if __name__=='__main__':
    main()