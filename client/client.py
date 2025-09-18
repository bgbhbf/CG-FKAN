import torch
from utils import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader  

class Client():
    def __init__(self, device, model_func, received_vecs, dataset, lr, args, grid_update=True):
        self.args = args
        self.device = device
        self.model_func = model_func
        self.prev_grid = None  # Initialize previous grid
        self.received_vecs = received_vecs
        self.comm_vecs = {
            'local_update_list': None,
            'local_model_param_list': None,
        }
        
        if self.received_vecs['Params_list'] is None:
            raise Exception("CommError: invalid vectors Params_list received")
        self.model = set_client_from_params(device=self.device, model=self.model_func(), params=self.received_vecs['Params_list'])

        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.local_learning_rate)
        self.dataset = dataset
        self.grid_update = grid_update
        self.batch_size = args.batchsize
        self.max_norm = 10

    def train(self):
        self.received_vecs['Params_list'] = self.received_vecs['Params_list'].to(self.device)
        train_input = torch.from_numpy(self.dataset[0]).float().to(self.device)
        train_label = torch.from_numpy(self.dataset[1]).float().to(self.device)

        loader = DataLoader(
            TensorDataset(train_input, train_label),
            batch_size=self.batch_size,
            shuffle=True
        )

        # 1) Grid가 바뀌었으면 매번 knot 재조정
        current_grid = self.args.grid
        if self.grid_update is False and self.args.model == 'kan':
            self.model.get_act(train_input)
            try:
                # MultKAN 계열
                self.model.update_grid_from_samples(train_input)
                # print(f"[Client] Updated grid from {self.prev_grid} to {current_grid}")
            except AttributeError:
                # efficient KANLinear 계열
                for layer in self.model.layers:
                    layer.update_grid(train_input)

        self.model.train()

        for epoch in range(self.args.local_epochs):
            self.optimizer.zero_grad()
            preds = self.model(train_input)
            loss  = torch.sqrt(F.mse_loss(preds, train_label))
            loss.backward()
            self.optimizer.step()

        last_state = get_mdl_params(self.model).to(self.device)
        self.comm_vecs['local_update_list']      = last_state - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state
        return self.comm_vecs, loss.item()
