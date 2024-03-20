
import torch

def calculate_ga_index(ga, size):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        increment = (40-20)/size
        ga_mapped = torch.round((ga - 20) / increment)
        return ga_mapped    
def create_bi_partitioned_ordinal_vector(gas, size):
        # Adjusting the threshold for the nearest 0.1 increment
        if  size is None or size <= 0 :
             return
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        ga_indices = calculate_ga_index(gas, size)
        vectors = torch.full((batch_size, size), -1, device=device)  # Default fill with -1

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > size:
                idx = size
            elif idx < 0:
                idx = 1
            
            if idx >= threshold_index:  # GA >= 30
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1  # First 100 elements to 1 (up to GA == 30)
                vectors[i, new_idx:] = 0  # The rest to 0
            else:  # GA < 30
                new_idx = idx*2
                vectors[i, :new_idx] = 0  # First 100 elements to 0
                # The rest are already set to -1

        return vectors