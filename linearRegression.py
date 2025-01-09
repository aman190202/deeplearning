import torch

class syntheticData():
    def __init__(self, w, b , train_num = 100, valid_num = 20,  test_num = 20, batch_size = 32, noise = 0.01):
        n_elements = train_num + valid_num + test_num
        self.X = torch.randn(n_elements,len(w)) # n len(w) sized vector 
        noise = torch.randn(n_elements, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
    def __len__(self):
        return self.X.shape, self.y.shape
    



dataset = syntheticData(torch.Tensor([2.0]),torch.Tensor([1.0]))
print(dataset.__len__())