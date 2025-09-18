import os
import time
import numpy as np

import torch
from utils import *
from dataset import Dataset
from torch.utils import data
import torch.nn.functional as F
# from efficientKAN import KAN
import math
from torch.utils.tensorboard import SummaryWriter


check_block_number =3
threshold1 = 197
threshold2 = 900

class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        self.max_vram = np.zeros((args.comm_rounds))
        self.server_model = init_model
        self.server_model_params_list = init_par_list 
        if self.args.sparsification == 1:
            tb_logdir = os.path.join('tb_logs', f"{args.dataset}_{self.args.problem_id}_{args.method}/LR{args.local_learning_rate}/layer_{self.args.layer}/SRX/quant{self.args.quantization_levels}/{args.model}_grid_CG-FKAN", args.out_file)
        else:
            tb_logdir = os.path.join('tb_logs', f"{args.dataset}_{self.args.problem_id}_{args.method}/LR{args.local_learning_rate}/layer_{self.args.layer}/SRX/quant{self.args.quantization_levels}/{args.model}_grid_{args.grid}_Fixed", args.out_file)
            if args.grid_varing is True:
                tb_logdir = os.path.join('tb_logs', f"{args.dataset}_{self.args.problem_id}_{args.method}/LR{args.local_learning_rate}/layer_{self.args.layer}/SRX/quant{self.args.quantization_levels}/{args.model}_grid_extension", args.out_file)
            

        self.writer = SummaryWriter(log_dir=tb_logdir)        
        self.tb_logdir = tb_logdir
        print("Initialize the Server      --->  {:s}".format(self.args.method))
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(self.clients_params_list.shape[0], self.clients_params_list.shape[1]))
        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        ### Generate Log Storage : [[loss, acc]...] * T
        self.test_perf = np.zeros((self.args.comm_rounds, 2))
        self.min_metric = LARGE_VALUE
        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate
        self.local_loss = LARGE_VALUE
        self.comm_vecs = {
            'Params_list': None,
        }
        self.received_vecs = None
        self.Client = None
        self._build_param_map()
        self.sparse_indices = None
        if self.args.grid_varing is True:
            self.client_grid_updated = [False] * self.args.total_client
        else:
            self.client_grid_updated = [True] * self.args.total_client      

        self.restricted_bits = args.restricted_bits                    

    def _get_block_grid_info(self, shape):
        if len(shape) == 3:
            total_len = shape[2]
            coef_len  = self.args.grid + self.args.spline_order
            act_len   = self.args.spline_order + 1

            if total_len == coef_len:
                return self.args.spline_order, self.args.grid
            elif total_len == act_len:
                # activation spline’s small buffer (rare)
                return 0, total_len
            else:
                return 0, total_len

        elif len(shape) == 2:
            return self.args.spline_order, self.args.grid
        else:
            raise ValueError(f"Unsupported shape in _get_block_grid_info: {shape}")

    def _build_param_map(self):
        model = self.model_func().to(self.device)
        flat = get_mdl_params(model)
        offset = 0
        self.param_slices = {}
        for name, param in model.named_parameters():
             n = param.numel()
             self.param_slices[name] = (offset, offset + n, param.shape)
             offset += n
        self.spline_ranges = [(name, s, e, shape) for name,(s,e,shape) in self.param_slices.items() if name.endswith("coef") or name.endswith("grid")]

    def _build_param_index_map(self):
        model = self.model_func().to(self.device)
        flat = get_mdl_params(model)
        offset = 0
        self.param_slices = {}
        for name, param in model.named_parameters():
            # print(f"name: {name}, param:{param}")
            n = param.numel()
            self.param_slices[name] = (offset, offset + n, param.shape)
            offset += n

        self.spline_ranges = [
            (s, e, shape) 
            for name, (s, e, shape) in self.param_slices.items()
            if name.endswith("spline_weight")
        ]
        print(f"[Server] spline ranges size: {self.spline_ranges}")
    
    def _activate_clients_(self, t):    
        num_clients = self.args.total_client
        k = max(int(self.args.active_ratio * num_clients), 1)
        rng = np.random.default_rng()
        return rng.choice(num_clients, size=k, replace=False).tolist()

            
    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay

    def _test_(self, t, selected_clients):
        loss, metric = self._validate_((self.datasets.test_x, self.datasets.test_y))
        self.test_perf[t] = [loss, metric]
        self.writer.add_scalar('Test/RMSE', metric, t)
        if self.min_metric > metric:
            self.min_metric = metric
            self.writer.add_scalar('MIN_RMSE', self.min_metric, t)
            # Save the best model
        print(f"    Test  ----  RMSE: {metric:.4e}, minRMSE: {self.min_metric:.4e}", flush=True)
    

    def _summary_(self):
        summary_root = self.tb_logdir             
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        summary_file = summary_root + f'/{self.args.dataset}_{self.args.model}_layer_{self.args.layer}_grid_{self.args.grid}_{self.args.method}.txt'
        with open(summary_file, 'w') as f:
            f.write("##=============================================##\n")
            f.write("##                   Summary                   ##\n")
            f.write("##=============================================##\n")
            f.write("Communication round   --->   T = {:d}\n".format(self.args.comm_rounds))
            f.write("Average Time / round   --->   {:.2f}s \n".format(np.mean(self.time)))
            best_rmse = np.min(self.test_perf[:,1])
            best_round = np.argmin(self.test_perf[:,1])
            f.write(f"Best Test RMSE (T)    --->   {best_rmse:.4e} ({best_round})\n")
        


    def _validate_(self, dataset):
        """
        Validate the server model on the given dataset.

        Args:
            dataset (tuple): A tuple (inputs, labels) where inputs and labels are numpy arrays.

        Returns:
            tuple: (avg_loss, metric)
                - For regression (Feynman): avg_loss is None, metric is RMSE over all samples.
                - For classification: avg_loss is average cross-entropy loss, metric is accuracy.
        """
        self.server_model.eval()
        test_input = torch.from_numpy(dataset[0]).float().to(self.device)
        test_label = torch.from_numpy(dataset[1]).float().to(self.device)

        with torch.no_grad():
            preds = self.server_model(test_input)
            # print(f"pred {preds}")
            mse   = F.mse_loss(preds, test_label, reduction='mean')
            mse   = torch.clamp(mse, min=1e-12)
            rmse  = torch.sqrt(mse)

        return None, rmse.item()


    def process_for_communication(self):
        pass
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass

    def fast_count_kept(self, full_update, spline_ranges, args):
        self._build_param_map()
        N = full_update.numel()
        print(N)

        L_ranges = []
        coef_len = args.grid + args.spline_order

        for name, s, e, shape in spline_ranges:
            if name.endswith("coef"):
                num_blocks = math.prod(shape[:-1])
                block_len  = shape[-1]
                for b in range(num_blocks):
                    start = s + b * block_len
                    end   = start + block_len
                    L_ranges.append((start, end, (1, block_len)))

            if name.endswith("grid") and not args.act_grid:
                num_blocks = shape[0]               
                block_len  = shape[-1]              
                for b in range(num_blocks):
                    start = s + b * block_len
                    end   = start + block_len
                    L_ranges.append((start, end, (1, block_len)))
        return L_ranges

    def postprocess(self, client, received_vecs, t):
        if self.args.grid_varing is True:
            self._build_param_map()
        full_update = received_vecs['local_update_list']
        N = full_update.numel()

        coef_len = self.args.grid + self.args.spline_order
        L_ranges = []
        for name, s, e, shape in self.spline_ranges:
            if name.endswith("coef"):
                if shape[-1] == coef_len:
                    L_ranges.append((s, e, shape))
            elif name.endswith("grid") and self.args.act_grid != True:
                L_ranges.append((s, e, shape))

        layer_node_counts = []
        for _, _, shape in L_ranges:
            if len(shape) == 3:
                oc, ic, _ = shape
            else:  # len(shape)==2 (.grid)
                oc, _ = shape
                ic = 1
            layer_node_counts.append(oc * ic)        
        total_blocks = sum(layer_node_counts)
        k_ratio = self.args.sparse_ratio

        block_id_to_layer = []
        for layer_idx, (_, _, shape) in enumerate(L_ranges):
            if len(shape) == 3:
                oc, ic, _ = shape
            else:
                oc, _ = shape
                ic = 1
            for _ in range(oc * ic):
                block_id_to_layer.append(layer_idx)        
        B = len(block_id_to_layer)

        drop_local = {b: [] for b in range(B)}

        alpha = self.args.sparse_alpha
        G_tot = max(int(round(1.0/alpha)), 1)
        groups = [[b] for b in range(B)]

        for grp in groups:
            grid_to_blocks = {}
            for b in grp:
                _, local_grid = self._get_block_grid_info(L_ranges[block_id_to_layer[b]][2])
                grid_to_blocks.setdefault(local_grid, []).append(b)
            for local_grid, blocks in grid_to_blocks.items():
                agg = None
                for b in blocks:
                    cum = 0
                    for s, e, shape in L_ranges:
                        if len(shape) == 3:
                            oc, ic, tot_len = shape
                        else:
                            oc, ic = shape[0], 1
                            tot_len = shape[1]
                        cnt = oc * ic
                        if b < cum + cnt:
                            sub = b - cum
                            base = s + sub * tot_len
                            order, _ = self._get_block_grid_info(shape)
                            slice_val = full_update[base + order : base + order + local_grid].abs()
                            break
                        cum += cnt                        
                    agg = slice_val.clone() if agg is None else agg + slice_val
                # k_drop = max(int(agg.numel()*k_ratio),1)
                k_drop = int(agg.numel()*k_ratio)
                if k_drop <= 0:
                    continue
                _, idxs = torch.topk(agg, k_drop, largest=False)
                to_drop = idxs.tolist()
                for b in blocks:
                    drop_local[b] = to_drop.copy()


        drop_idxs_all = []
        for b, locals_idx in drop_local.items():
            cum = 0
            for s, e, shape in L_ranges:
                if len(shape) == 3:
                    oc, ic, tot_len = shape
                else:
                    oc, ic = shape[0], 1
                    tot_len = shape[1]
                cnt = oc * ic
                if b < cum + cnt:
                    sub = b - cum
                    base = s + sub * tot_len
                    order, _ = self._get_block_grid_info(shape)
                    for i_local in locals_idx:
                        drop_idxs_all.append(base + order + i_local)
                    break
                cum += cnt

        keep = sorted(set(range(N))-set(drop_idxs_all))
        masked = torch.zeros_like(full_update); masked[keep]=full_update[keep]
        received_vecs['local_update_list']=masked
        received_vecs['sparse_indices']=torch.tensor(keep,dtype=torch.int32)
        received_vecs['sparse_values']=masked[keep]


        dummy = 0
        for idx in range(0,check_block_number):
            print(f"[Postprocess] idx: {idx}, L_ranges: {L_ranges[idx]}")
            start, end, shape = L_ranges[idx]
            if len(shape) == 2:
                dummy = dummy + round((end-start)/(self.args.spline_order*2 + self.args.grid+1))*k_drop
            elif len(shape) == 3:
                dummy = dummy + round((end-start)/(self.args.spline_order + self.args.grid))*k_drop
            else:
                raise ValueError(f"Unsupported shape length: {len(shape)} for shape {shape}")
            print(f"shape: {shape}, shapelen: {len(shape)} start: {start}, end: {end}, dummy: {dummy}")

        array = drop_idxs_all[dummy:k_drop+dummy]
        masked = full_update.clone()
        masked[array] = 0

        colombo = math.ceil(total_blocks * math.log2(math.comb(self.args.grid, math.ceil(self.args.grid*self.args.sparse_ratio))))
        transmitted_bts = len(keep) * float_bits + colombo
        self.writer.add_scalar('required_bits', transmitted_bts, t)
    
    
    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        if self.args.sparsification == 1 and self.args.sparse_method == 'fixed':
            self.sparse_indices = np.random.choice(range(self.args.grid), max(int(self.args.grid*self.args.sparse_ratio), 1), replace=False) #+ self.args.spline_order       
        return_round = 0
        for t in range(self.args.comm_rounds):
            update_round = (self.args.comm_rounds/5)
            # print((t % update_round))
            if self.args.grid_varing is True and t > 0 and (t % update_round) > threshold1 and (t<threshold2): #only at >=30
                self.client_grid_updated = [False] * self.args.total_client
                print(f"*** PRevious Round {t}: grid updated to {self.args.grid} ***")
            elif self.args.grid_varing is True and t > 0 and (t % update_round) == 0:
                delta_grid = [0, 2, 5, 20, 20, 50, 100]
                new_grid = self.args.grid + delta_grid[round(t/update_round)]
                if self.args.sparse_varing is True and new_grid > 9:
                    if new_grid == 10:
                        self.args.sparse_ratio = 0.1
                    elif new_grid == 30:
                        self.args.sparse_ratio = 0.7
                    elif new_grid == 50:    
                        self.args.sparse_ratio = 0.8
                    elif new_grid == 100:    
                        self.args.sparse_ratio = 0.9
                self.writer.add_scalar('GRID_VARIED', new_grid, t)
                print(f"*** Round {t}: manual grid {self.args.grid} → {new_grid} ***")
                self.server_model = self.server_model.refine(new_grid)
                self.args.grid = new_grid
                self.client_grid_updated = [False] * self.args.total_client

                import copy
                self.model_func = lambda: copy.deepcopy(self.server_model)

                # 7) 파라미터(flatten)·버퍼 재초기화
                flat = torch.cat([p.data.reshape(-1) for p in self.server_model.parameters()], dim=0)
                self.server_model_params_list = flat.clone()
                self.comm_vecs['Params_list'] = flat.clone()
                num_clients = self.args.total_client
                device, dtype = flat.device, flat.dtype
                self.clients_params_list         = torch.zeros((num_clients, flat.numel()), device=device, dtype=dtype)
                self.clients_updated_params_list = torch.zeros_like(self.clients_params_list)
                Averaged_update                  = torch.zeros_like(flat, device=device)            

                model_temp = self.model_func().to(self.device)
                total_weight = 0
                for name, param in model_temp.named_parameters():
                    total_weight += param.numel()

                total_bits = total_weight*float_bits
                print(f"Total weight: {total_weight}, Total bits: {total_bits}")
                print(flat.shape, flat.numel(), "flat numel")
                if self.args.sparse_varing and new_grid > 9:
                    num_kept     = len(self.fast_count_kept(flat, self.spline_ranges, self.args))
                    total_weight = flat.numel()
                    G            = self.args.grid
                    final_ratio  = None

                    for i in range(0, 20):
                        temp_spars_ratio = i / 20.0
                        k = math.ceil(G * temp_spars_ratio)
                        colombo = math.ceil(num_kept * math.log2(math.comb(G, k)))
                        current_bits = (total_weight- num_kept * G * temp_spars_ratio) * float_bits + colombo

                        if current_bits <= self.restricted_bits:
                            final_ratio = temp_spars_ratio
                            break
                        final_ratio = temp_spars_ratio

                    if final_ratio is None:
                        final_ratio = 0.0
                    self.args.sparse_ratio = final_ratio
                self.writer.add_scalar('Sparsification_ratio', self.args.sparse_ratio, t)

            start = time.time()
            # select active clients list
            selected_clients = self._activate_clients_(t)
            counts = torch.tensor([len(self.datasets.client_y[c]) for c in selected_clients],device=self.device,dtype=torch.float32)
            weights = counts / counts.sum()  # 합이 1이 되도록 정규화
            weights_u = weights.unsqueeze(1)   # shape: [K,1]
            
            print('============= Communication Round', t + 1,'SR',self.args.sparse_ratio,'=============', flush = True)
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
            if self.args.sparsification == 1 and self.args.sparse_method == 'random':
                self.sparse_indices = np.random.choice(range(self.args.grid), max(int(self.args.grid*self.args.sparse_ratio), 1), replace=False)#+ self.args.spline_order
            temp_local_loss = 0
            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs, dataset=dataset, lr=self.lr, args=self.args, grid_update=self.client_grid_updated[client])
                self.client_grid_updated[client] = True
                self.received_vecs, loss = _edge_device.train()
                temp_local_loss += loss
                if self.args.sparsification == 1 and self.args.grid>9:
                    self.postprocess(client, self.received_vecs, t)
                else:
                    model_temp = self.model_func().to(self.device)
                    total_weight = 0
                    for name, param in model_temp.named_parameters():
                        total_weight += param.numel()
                    self.writer.add_scalar('number_weight', total_weight, t)
                    self.writer.add_scalar('required_bits', total_weight*float_bits, t)
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                # release the salloc
                del _edge_device
            
            if self.args.non_iid is False:
                Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
                Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            else:
                updates = self.clients_updated_params_list[selected_clients]  # shape: [K,D]
                weights_u = weights_u.to(updates.device)
                models  = self.clients_params_list[selected_clients]          # shape: [K,D]
                Averaged_update = (updates * weights_u).sum(dim=0)  # [D]
                Averaged_model  = (models  * weights_u).sum(dim=0)  # [D]

            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            end = time.time()
            self.time[t] = end-start
            print("            ----    Time: {:.2f}s".format(self.time[t]))    
            
        self._summary_()
        self.writer.close()
