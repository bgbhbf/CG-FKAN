import torch

LARGE_VALUE = 1e+10
float_bits = 32
train_data_num = 50000
test_data_num = 10000

def get_mdl_params(model):
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)

def param_to_vector(model):
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)
    
def set_client_from_params(device, model, params):
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)

def get_params_list_with_shape(model, param_list, device):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape).to(device))
        idx += length
    return vec_with_shape


def quantize(x, input_compress_settings={}):
    """    
    Inputs:
    - x: torch tensor
    - input_compress_settings: dict, 'n'
    
    Outputs:
    - Tilde_x: quantized tensor
    """
    compress_settings = {'n': 6}  
    compress_settings.update(input_compress_settings)
    
    n = compress_settings['n']
    x = x.float()
    x_norm = torch.norm(x, p=float('inf'))
    sgn_x = ((x > 0).float() - 0.5) * 2
    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p  
    margin = (compare < final_p).float() 
    xi = (floor_p + margin) / n
    Tilde_x = x_norm * sgn_x * xi
    
    return Tilde_x


def calculate_kan_weights(spline_order: int, grid: int,
                          hidden_configs_layer: list,
                          len_input_vars: int):
    """
    The total number of spline coefficients in a KAN model.
    """
    layers = [len_input_vars] + hidden_configs_layer + [1]
    total_coeffs = 0
    G = grid + spline_order
    
    for i in range(len(layers)-1):
        in_dim  = layers[i]
        out_dim = layers[i+1]
        cnt = out_dim * in_dim * G
        total_coeffs += cnt
    
    return total_coeffs


def calculate_mlp_multi_hidden_layers(target_weights: int, input_dim: int, num_classes: int = 1, num_hidden_layers: int = 2):
    """
    To match a target number of weights in an MLP with multiple hidden layers.
    """
    if num_hidden_layers == 1:
        h = (target_weights - num_classes) / (input_dim + 1 + num_classes)
        hidden_size = max(1, int(round(h)))
        return [hidden_size]
    
    if num_hidden_layers == 2:
        total_for_h = target_weights - num_classes
        h1 = int(round(total_for_h ** 0.5)) 
        h2 = max(1, int(round(h1 * 0.6)))  
        
        actual_weights = input_dim * h1 + h1 + h1 * h2 + h2 + h2 * num_classes + num_classes
        while actual_weights > target_weights and h1 > 1:
            h1 -= 1
            h2 = max(1, int(round(h1 * 0.6)))
            actual_weights = input_dim * h1 + h1 + h1 * h2 + h2 + h2 * num_classes + num_classes
        
        return [h1, h2]
    
    h = max(1, int(round((target_weights / num_hidden_layers) ** 0.5)))
    return [h] * num_hidden_layers