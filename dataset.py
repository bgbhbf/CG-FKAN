
import numpy as np
import torch
from utils import *
from feynman import get_feynman_dataset
from kan.utils import create_dataset

        
class DatasetObject:
    def __init__(self, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', args=None):
        self.args = args
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        dataset_name_lower = self.args.dataset.lower()
        self.name = f"{dataset_name_lower}_{self.n_client}_{self.seed}_{self.rule}_{rule_arg_str}"
        self.name += f"_{unbalanced_sgm:f}" if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
       
    def set_data(self):  
        problem_id = int(self.args.problem_id)                                      
        input_vars, _, func, ranges = get_feynman_dataset(problem_id)       

        n_var=len(input_vars)
        rng_state = np.random.get_state()
        ds = create_dataset(                                                       
            f=func,                                                                
            n_var=n_var,
            train_num=train_data_num, 
            test_num=test_data_num,
            ranges=ranges,
            device=self.args.device,                                              
            seed=self.seed                                                        
        )                   

        np.random.set_state(rng_state)
        train_x = ds['train_input'].cpu().numpy()                                  
        train_y = ds['train_label'].cpu().numpy()                                  
        test_x  = ds['test_input'].cpu().numpy()                                   
        test_y  = ds['test_label'].cpu().numpy()               

        # Feynman은 회귀 문제이므로 채널/크기 세팅을 다르게 함
        self.channels = n_var                                                      
        self.width    = 1                                                          
        self.height   = 1                                                          
        self.n_cls    = train_y.shape[1]                               
            
        n_data_per_client = int(len(train_y) / self.n_client)
        client_data_list = np.ones(self.n_client, dtype=int) * n_data_per_client
        diff = np.sum(client_data_list) - len(train_y)
        
        if diff != 0:
            for client_i in range(self.n_client):
                if client_data_list[client_i] > diff:
                    client_data_list[client_i] -= diff
                    break

        if self.rule == 'Quantity_skew':
            num_samples = len(train_x)        
            idxs = np.random.permutation(num_samples)
            min_size = 0
            while min_size < 10:
                proportions = np.random.dirichlet(np.repeat(self.args.split_coef, self.n_client))
                proportions = proportions/proportions.sum()
                min_size = np.min(proportions*len(idxs))
                expected_counts = proportions * num_samples
                min_size = expected_counts.min()

            split_points = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
            batch_idxs = np.split(idxs, split_points) 
            client_x = [np.zeros((len(batch_idxs[i]), self.channels, self.height, self.width), dtype=np.float32)
                        for i in range(self.n_client)]
            client_y = [np.zeros((len(batch_idxs[i]), 1), dtype=np.float32)  # regression이니 float32
                        for i in range(self.n_client)]
            for i in range(self.n_client):
                idx_i = batch_idxs[i]
                client_x[i] = train_x[idx_i]
                client_y[i] = train_y[idx_i].reshape(-1, 1)
            self.client_x = client_x
            self.client_y = client_y    

        elif self.rule == 'iid':
            client_x = [np.zeros((client_data_list[client__], self.channels, self.height, self.width), dtype=np.float32)
                        for client__ in range(self.n_client)]
            client_y = [np.zeros((client_data_list[client__], 1), dtype=np.int64)
                        for client__ in range(self.n_client)]
            client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
            for client_idx_ in range(self.n_client):
                client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
            self.client_x = client_x
            self.client_y = client_y

        self.test_x = test_x
        self.test_y = test_y
            
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y=True, train=False, dataset_name='', args=None):
        self.name = dataset_name
        self.args = args
        self.train = train
        
        self.X_data = torch.from_numpy(data_x).float()   
        self.y_data = torch.from_numpy(data_y).float()   
        self.channels = data_x.shape[1]                  
        self.width    = 1                                
        self.height   = 1                                
        self.n_cls    = data_y.shape[1]                  
        
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x = self.X_data[idx]
        inp = x.clone()
        y = self.y_data[idx]
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        return inp, y        


