import pickle
import struct
import numpy as np
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
import sys
from config import change_config
from config import get_params

if len(sys.argv) > 1:
    change_config(sys.argv[1])

path_raw_graph, path_compressed_graph, svd_treshold, spectral_eigs, kmeans_klusters, window_size = get_params(
        "path_raw_graph", "path_compressed_graph", "svd_treshold", "spectral_eigs", "kmeans_klusters", "window_size")

class _VVT_Operator(spla.LinearOperator):
    """Linear operator used to perform implicit iteration of Lanczos algorithm,
    reducing matrix needed to be stored in RAM"""
    def __init__(self, V):
        self.V = V
        super().__init__(V.dtype, (V.shape[0], V.shape[0]))

    def _matvec(self, x):
        return self.V @ (self.V.T @ x)

def encode_vbyte(value):
    if value == 0:
        return b'\x00'
    b = bytearray()
    while value > 0:
        byte = value & 0x7F
        value >>= 7
        if value > 0:
            byte |= 0x80 #continues further?
        b.append(byte)
    return bytes(b)

def compress_graph(path_raw_graph = path_raw_graph, 
                   path_compressed_graph = path_compressed_graph, 
                   svd_treshold = svd_treshold, 
                   spectral_eigs = spectral_eigs,
                   kmeans_klusters = kmeans_klusters, 
                   window_size = window_size):
    print("Loading raw graph from disk...")
    with open(path_raw_graph, "rb") as f:
        data = pickle.load(f)
        A = data["matrix"]
        node_to_idx = data["node_to_idx"]
        
    N = A.shape[0]

    out_degrees = np.array(A.sum(axis=1)).flatten()
    out_degrees[out_degrees == 0] = 1.0  

    A_T = A.T.tocsr() 

    print("Computing RSVD...")
    k_svd = min(svd_treshold, N - 1)
    U, s, Vt = spla.svds(A_T.astype(float), k=k_svd)
    V_clean = Vt.T * s  

    print("Performing Implicit Lanczos on V*V^T...")
    vvT_op = _VVT_Operator(V_clean)
    eigenvals, eigenvecs = spla.eigsh(vvT_op, k=spectral_eigs, which='LM')

    print("Clustering nodes for compression ordering...")
    kmeans = KMeans(n_clusters=kmeans_klusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(eigenvecs)

    new_order = np.argsort(labels)
    new_to_old = {new_idx: old_idx for new_idx, old_idx in enumerate(new_order)}

    print("Permuting Adjacency Matrix to Spectral Order...")
    A_T_ordered = A_T[new_order, :]      
    A_T_ordered = A_T_ordered[:, new_order]  

    print("Applying Diff-Based Reference VByte Encoding...")
    window_cache = [] 
    
    with open(path_compressed_graph, "wb") as f:
        # 1. Write out degrees
        f.write(struct.pack("<I", N))
        ordered_out_degrees = np.zeros(N, dtype=np.float32)
        for new_idx in range(N):
            ordered_out_degrees[new_idx] = out_degrees[new_to_old[new_idx]]
        f.write(ordered_out_degrees.tobytes())
    
        # 2. Row compression...
        reference_count = 0 #collect statistics
        for i in range(N):
            reference_choosen = False
            row_links = A_T_ordered.indices[A_T_ordered.indptr[i]:A_T_ordered.indptr[i+1]]
            row_links = np.sort(row_links)

            #No reference
            raw_deltas = np.diff(row_links).tolist() if len(row_links) > 0 else []
            raw_first = int(row_links[0]) if len(row_links) > 0 else -1
            best_cost = 1 + len(raw_deltas) 
            
            best_encoding = {
                "ref_offset": 0,
                "first": raw_first,
                "deltas": [int(d) for d in raw_deltas]
            }

            #Yes reference
            for offset_idx, prev_links in enumerate(reversed(window_cache)):
                offset = offset_idx + 1

                to_add = np.setdiff1d(row_links, prev_links)
                to_remove = np.setdiff1d(prev_links, row_links)

                add_deltas = np.diff(to_add).tolist() if len(to_add) > 0 else []
                add_first = int(to_add[0]) if len(to_add) > 0 else -1
                
                remove_deltas = np.diff(to_remove).tolist() if len(to_remove) > 0 else []
                remove_first = int(to_remove[0]) if len(to_remove) > 0 else -1

                cost_ref = 1 + len(add_deltas) + len(remove_deltas)

                if cost_ref < best_cost:
                    reference_choosen = True
                    best_cost = cost_ref
                    best_encoding = {
                        "ref_offset": offset,
                        "add_first": add_first,
                        "add_deltas": [int(d) for d in add_deltas],
                        "add_count": len(to_add),
                        "remove_first": remove_first,
                        "remove_deltas": [int(d) for d in remove_deltas],
                        "remove_count": len(to_remove)
                    }

            if reference_choosen: reference_count += 1

            #writing to the file...
            if best_encoding["ref_offset"] > 0:
                f.write(struct.pack("<B", best_encoding["ref_offset"]))
                
                f.write(encode_vbyte(best_encoding["remove_count"]))
                if best_encoding["remove_count"] > 0:
                    f.write(encode_vbyte(best_encoding["remove_first"]))
                    for d in best_encoding["remove_deltas"]:
                        f.write(encode_vbyte(d))
                        
                f.write(encode_vbyte(best_encoding["add_count"]))
                if best_encoding["add_count"] > 0:
                    f.write(encode_vbyte(best_encoding["add_first"]))
                    for d in best_encoding["add_deltas"]:
                        f.write(encode_vbyte(d))
                        
            elif best_encoding["first"] == -1:
                f.write(struct.pack("<B", 0))
            else:
                f.write(struct.pack("<B", 255))
                f.write(encode_vbyte(best_encoding["first"]))
                f.write(encode_vbyte(len(best_encoding["deltas"])))
                for d in best_encoding["deltas"]:
                    f.write(encode_vbyte(d))

            window_cache.append(row_links)
            if len(window_cache) > window_size:
                window_cache.pop(0)

    print("Writing metadata dictionary...")
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
    with open("graph_metadata.pkl", "wb") as f:
        pickle.dump({"idx_to_node": idx_to_node, "new_to_old": new_to_old}, f)
        
    print(f"Compression complete with {reference_count} references performed!")

if __name__ == "__main__":
    compress_graph()
