import pickle
import numpy as np
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
import config
import sys

class _VVT_Operator(spla.LinearOperator):
    """Linear operator used to perform implicit iteration of Lanczos alghoritm,
    reducing matrix needed to be stored in RAM"""
    def __init__(self, V):
        self.V = V
        super().__init__(V.dtype, (V.shape[0], V.shape[0]))

    def _matvec(self, x):
        return self.V @ (self.V.T @ x)

def compress_graph(path_raw_graph, svd_treshold, 
                   spectral_eigs, kmeans_klusters, window_size):
    print("Loading raw graph from disk...")
    with open(path_raw_graph, "rb") as f:
        data = pickle.load(f)
        A = data["matrix"]
        node_to_idx = data["node_to_idx"]
        
    N = A.shape[0]

    out_degrees = np.array(A.sum(axis=1)).flatten()
    out_degrees[out_degrees == 0] = 1.0  

    #we will compress transposed matrix for its future use in computation of PageRank
    A_T = A.T.tocsr() 

    # --- SVD & DIMENSIONALITY REDUCTION ---
    print("Computing RSVD...")
    k_svd = min(svd_treshold, N - 1)
    U, s, Vt = spla.svds(A_T.astype(float), k=k_svd)
    V_clean = Vt.T * s  

    # --- IMPLICIT LANCZOS ON VV^T ---
    print("Performing Implicit Lanczos on V*V^T...")
    vvT_op = _VVT_Operator(V_clean)
    eigenvals, eigenvecs = spla.eigsh(vvT_op, k=spectral_eigs, which='LM')

    # --- SPECTRAL CLUSTERING & REORDERING ---
    print("Clustering nodes for compression ordering...")
    kmeans = KMeans(n_clusters=kmeans_klusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(eigenvecs)

    new_order = np.argsort(labels)
    new_to_old = {new_idx: old_idx for new_idx, old_idx in enumerate(new_order)}

    print("Permuting Adjacency Matrix to Spectral Order...")
    A_T_ordered = A_T[new_order, :]      
    A_T_ordered = A_T_ordered[:, new_order]  

    # --- REFERENCE COMPRESSION ---
    print("Applying Diff-Based Reference Encoding...")
    compressed_rows = []
    window_cache = [] 
    
    for i in range(N):
        row_links = A_T_ordered.indices[A_T_ordered.indptr[i]:A_T_ordered.indptr[i+1]]
        row_links = np.sort(row_links)

        raw_deltas = np.diff(row_links).tolist() if len(row_links) > 0 else []
        raw_first = int(row_links[0]) if len(row_links) > 0 else -1
        
        best_cost = 1 + len(raw_deltas) 
        best_encoding = {
            "ref_offset": 0,
            "first": raw_first,
            "deltas": [int(d) for d in raw_deltas]
        }

        for offset_idx, prev_links in enumerate(reversed(window_cache)):
            offset = offset_idx + 1

            to_add = np.setdiff1d(row_links, prev_links)
            to_remove = np.setdiff1d(prev_links, row_links)

            add_deltas = np.diff(to_add).tolist() if len(to_add) > 0 else []
            add_first = int(to_add[0]) if len(to_add) > 0 else -1

            cost_ref = 1 + len(add_deltas) + len(to_remove)

            if cost_ref < best_cost:
                best_cost = cost_ref
                best_encoding = {
                    "ref_offset": offset,
                    "add_first": add_first,
                    "add_deltas": [int(d) for d in add_deltas],
                    "remove": [int(r) for r in to_remove]
                }

        compressed_rows.append(best_encoding)

        window_cache.append(row_links)
        if len(window_cache) > window_size:
            window_cache.pop(0)

    print("Writing compressed graph and metadata to disk...")
    ordered_out_degrees = np.zeros(N)
    for new_idx in range(N):
        ordered_out_degrees[new_idx] = out_degrees[new_to_old[new_idx]]

    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
    
    with open("compressed_graph.pkl", "wb") as f:
        pickle.dump({
            "N": N,
            "out_degrees": ordered_out_degrees,
            "idx_to_node": idx_to_node,      
            "new_to_old": new_to_old,        
            "compressed_rows": compressed_rows
        }, f)
        
    print("Compression Complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.change_config(sys.argv[1])
    compress_graph(**config.get_compress_params())
