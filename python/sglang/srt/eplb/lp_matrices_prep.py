from typing import List, Optional
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack
from dataclasses import dataclass
import time

@dataclass
class TokenDispatchMetadata:
    ecos_opts: dict
    dims: dict
    B1: List[csc_matrix]
    B2: List[csc_matrix]
    C: List[csc_matrix]
    c: List[np.ndarray]
    G: List[csc_matrix]
    A: List[csc_matrix]
    single_expert_array: List[np.ndarray]
    log_replicated_expert_array: List[np.ndarray]
    phy_replicated_expert_array: List[np.ndarray]

    @staticmethod
    def init(phy2log, log2phy, g):
        assert phy2log.shape[0] == log2phy.shape[0], "phy2log, log2phy must have the same number of layers"
        assert phy2log.shape[1] % g == 0, "Number of physical experts must be divisible by number of GPUs"
        num_layer = phy2log.shape[0]
        num_logical = log2phy.shape[1]
        logcnt = np.zeros((num_layer, num_logical), dtype=int)
        for layer_id in range(num_layer):
            logcnt[layer_id] = np.bincount(phy2log[layer_id], minlength=num_logical)

        B1 = []
        B2 = []
        C = []
        c = []
        G = []
        A = []
        single_expert_array = []
        log_replicated_expert_array = []
        phy_replicated_expert_array = []
        for layer_id in range(num_layer):
            (B1_layer, B2_layer, C_layer, 
             c_layer, G_layer, A_layer, 
             single_expert_array_layer, 
             log_replicated_expert_array_layer, 
             phy_replicated_expert_array_layer) = TokenDispatchMetadata.init_single_layer(phy2log[layer_id], logcnt[layer_id], g)
            B1.append(B1_layer)
            B2.append(B2_layer)
            C.append(C_layer)
            c.append(c_layer)
            G.append(G_layer)
            A.append(A_layer)
            single_expert_array.append(single_expert_array_layer)
            log_replicated_expert_array.append(log_replicated_expert_array_layer)
            phy_replicated_expert_array.append(phy_replicated_expert_array_layer)
        dims = {
            'l': G[0].shape[0],
            'q': [],
            'e': 0
        }
        ecos_opts = {
            'max_iters': 100,
            'abstol': 3e-2,
            'feastol': 1e-4,
            'verbose': False
        }
        return TokenDispatchMetadata(ecos_opts, dims, B1, B2, C, c, G, A, single_expert_array, log_replicated_expert_array, phy_replicated_expert_array)
        
    
    @staticmethod
    def init_single_layer(layer_phy2log, layer_logcnt, g):
        num_phy: int = layer_phy2log.shape[0]
        num_phy_gpu: int = num_phy // g

        single_expert_array: np.ndarray = np.argwhere(layer_logcnt == 1).ravel()
        log_replicated_expert_array: np.ndarray = np.argwhere(layer_logcnt > 1).ravel()
        phy_replicated_expert_array: np.ndarray = np.argwhere(layer_logcnt[layer_phy2log] > 1).ravel()

        single_expert_count: int = len(single_expert_array)
        log_replicated_expert_count: int = len(log_replicated_expert_array)
        phy_replicated_expert_count: int = len(phy_replicated_expert_array)

        B = np.zeros((g, num_phy))
        for i in range(g):
            B[i, i*num_phy_gpu:(i+1)*num_phy_gpu] = 1
        B1 = csc_matrix(B[:, single_expert_array])
        B2 = csc_matrix(B[:, phy_replicated_expert_array])

        C = lil_matrix((log_replicated_expert_count, phy_replicated_expert_count), dtype=np.float64)
        phy2log_rep = layer_phy2log[phy_replicated_expert_array]
        for i in range(log_replicated_expert_count):
            C[i, phy2log_rep==log_replicated_expert_array[i]] = 1.0
        C = csc_matrix(C)
        A = hstack([csc_matrix((C.shape[0], 1)), C])

        assert single_expert_count + phy_replicated_expert_count == num_phy, f"Total expert count must equal to num_phy={num_phy}"

        c = np.zeros(phy_replicated_expert_count+1)
        c[0] = 1.0

        row_indices = np.arange(g)
        col_indices = np.zeros(g, dtype=int)
        data_values = -np.ones(g)

        G1_left = csc_matrix((data_values, (row_indices, col_indices)), shape=(g, 1))
        G1 = hstack([G1_left, B2])
        G2 = hstack([csc_matrix((phy_replicated_expert_count, 1)), -1 * csc_matrix(np.eye(phy_replicated_expert_count))])
        G = vstack([G1, G2])
        G = csc_matrix(G)

        return B1, B2, C, c, G, A, single_expert_array, log_replicated_expert_array, phy_replicated_expert_array

_global_token_dispatch_metadata: Optional[TokenDispatchMetadata] = None


def get_global_token_dispatch_metadata():
    return _global_token_dispatch_metadata


def set_global_token_dispatch_metadata(value):
    global _global_token_dispatch_metadata
    assert _global_token_dispatch_metadata is None
    _global_token_dispatch_metadata = value


def test():
    layer_phy2log = np.tile(np.array(list(range(256))+list(range(16))), (58, 1))
    layer_logcnt = np.tile(np.array([2]*16+[1]*240), (58, 1))
    layer_log2phy = np.tile(np.array([[i, i+256] for i in range(16)] + [[i, -1] for i in range(16, 256)]), (58, 1, 1))
    g = 8
    
    start_time = time.perf_counter()
    for i in range(10):
        metadata = TokenDispatchMetadata.init(layer_phy2log, layer_log2phy, g)
    end_time = time.perf_counter()
    print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
    print(f"Time per iteration: {(end_time - start_time)*1000/10:.2f} ms")

if __name__ == "__main__":
    test()