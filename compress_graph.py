import pickle
import numpy as np
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans

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

    # --- DELTA & REFERENCE COMPRESSION ---
    print("Applying Delta and Reference Encoding...")
    compressed_rows = []
    
    # Stores tuples of (first_val, deltas) for the last W rows
    window_cache = [] 
    
    for i in range(N):
        row_links = A_T_ordered.indices[A_T_ordered.indptr[i]:A_T_ordered.indptr[i+1]]
        row_links = np.sort(row_links)

        deltas = np.diff(row_links).tolist() if len(row_links) > 0 else []
        first_val = row_links[0] if len(row_links) > 0 else -1

        #Comparing current row to previous window_size rows 
        matched_offset = 0
        for offset_idx, (prev_first, prev_deltas) in enumerate(reversed(window_cache)):
            if first_val == prev_first and deltas == prev_deltas:
                matched_offset = offset_idx + 1
                break

        if matched_offset > 0:
            compressed_rows.append({"ref_offset": matched_offset})
        else:
            compressed_rows.append({
                "ref_offset": 0,
                "first": int(first_val),
                "deltas": [int(d) for d in deltas]
            })

        # Update the sliding window cache
        window_cache.append((first_val, deltas))
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

import config
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.change_config(sys.argv[1])
    compress_graph(**config.get_compress_params())
