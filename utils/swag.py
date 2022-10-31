import torch

class SWAG:
    def __init__(self, base_model):
        self.base_model = base_model
        self.num_parameters = sum(param.numel() for param in self.base_model.parameters())
        self.var_clamp = 1e-30
        self.n = 0
        self.mean = self.flatten([param.detach().cpu() for param in self.base_model.parameters()])
        self.sq_mean = self.mean**2
        self.max_rank = 20
        self.rank = 0
        self.model_device = None
        self.cov_mat_sqrt = torch.unsqueeze(torch.zeros(self.num_parameters),0)

    def set_weights(self, model, vector, device=None):
        offset = 0
        for param in model.parameters():
            param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
            offset += param.numel()


    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sigma_diag, self.var_clamp)
        return self.mean, variance


    def sample(self, update_model, scale=0.5, diag_noise=True):
        mean, variance = self._get_mean_and_variance()

        self.cov_factor = self.cov_mat_sqrt.clone() / (self.cov_mat_sqrt.size(0) - 1) ** 0.5  # 1/(sqrt(K-1)) x D : K x n_param

        eps_low_rank = torch.randn(self.cov_factor.size()[0]) # z2 (size K)
        z = self.cov_factor.t() @ eps_low_rank # n_param (1 x n_param?)

        if diag_noise:
            z += variance.sqrt() * torch.rand(variance.size())  # sqrt(sigma_diag) x z1 (size n_param)
        z *= scale ** 0.5  # 1/sqrt(2)
        sample = mean + z

        # apply to parameters
        self.set_weights(update_model, sample, self.model_device)
        return sample


    def collect_model(self, base_model):
        w = self.flatten([param.detach().cpu() for param in base_model.parameters()])
        # first moment
        self.mean.mul_(self.n / (self.n + 1.0))
        self.mean.add_(w / (self.n + 1.0))

        # second moment
        self.sq_mean.mul_(self.n / (self.n + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n + 1.0))
        self.n += 1

        dev_vector = w - self.mean

        if self.rank + 1 > self.max_rank:
            self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, dev_vector.view(1, -1)), dim=0) # cov_mat_sqrt size K x n_param        
        self.rank = min(self.rank+1, self.max_rank)
        self.sigma_diag = self.sq_mean - self.mean**2
        #print(f'collect_model cov size {self.cov_mat_sqrt.shape}')

    def flatten(self, lst):
        tmp = [i.contiguous().view(-1,1) for i in lst]
        return torch.cat(tmp).view(-1)


    def unflatten_like(self, vector, likeTensorList):
        # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
        #    shaped like likeTensorList
        outList = []
        i=0
        for tensor in likeTensorList:
            #n = module._parameters[name].numel()
            n = tensor.numel()
            outList.append(vector[:,i:i+n].view(tensor.shape))
            i+=n
        return outList