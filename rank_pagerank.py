import pickle
import numpy as np

def decode_row(i, compressed_rows, decompressed_cache):
    """Reconstructs the target links using Reference & Delta decoding."""
    row_data = compressed_rows[i]
    
    # 1. Reference Decoding (Copy from referenced row)
    if row_data.get("ref_offset", 0) > 0:
        return decompressed_cache[i - row_data["ref_offset"]]
    
    # 2. Delta Decoding (Reconstruct absolute indices from integer gaps)
    first = row_data["first"]
    if first == -1:
        links = []
    else:
        deltas = row_data["deltas"]
        links = [first]
        current = first
        for d in deltas:
            current += d
            links.append(current)
            
    decompressed_cache[i] = links
    return links

def rank_pagerank(path_compressed_graph, iterations, damping):
    print("Loading graph metadata...")
    with open(path_compressed_graph, "rb") as f:
        data = pickle.load(f)
        
    N = data["N"]
    out_degrees = data["out_degrees"]
    compressed_rows = data["compressed_rows"]
    new_to_old = data["new_to_old"]
    idx_to_node = data["idx_to_node"]

    # Initialize PageRank arrays in cache
    pr_current = np.ones(N) / N
    pr_next = np.zeros(N)
    
    # Cache for reference decoding (only need to remember referenced window)
    decompressed_cache = {}

    print(f"Beginning {iterations} iterations of Streaming PageRank...")
    for it in range(iterations):
        pr_next.fill(0.0)
        
        # Iterate over the graph, decoding one row at a time
        for i in range(N):
            in_links = decode_row(i, compressed_rows, decompressed_cache)
            
            # PULL PageRank calculation
            score = 0.0
            for j in in_links:
                score += pr_current[j] / out_degrees[j]
                
            pr_next[i] = score
            
            if i >= 7 and (i - 7) in decompressed_cache:
                del decompressed_cache[i - 7]

        # Apply damping factor and account for dangling nodes
        dangling_sum = np.sum(pr_current[out_degrees == 1.0]) #0's were padded by 1's in the first phase
        base_score = (1.0 - damping) / N + damping * (dangling_sum / N)
        
        pr_current = base_score + (damping * pr_next)
        
        print(f"  Iteration {it + 1}/{iterations} completed.")

    print("\nPageRank calculation finished!")
    
    top_10_new_idx = np.argsort(pr_current)[::-1][:10]
    
    print("\n--- TOP 10 WIKIPEDIA PAGE IDs ---")
    for rank, new_idx in enumerate(top_10_new_idx):
        score = pr_current[new_idx]
        old_idx = new_to_old[new_idx]
        real_page_id = idx_to_node[old_idx]
        print(f"#{rank + 1}: Page ID {real_page_id} (Score: {score:.6f})")

import sys
import config

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.change_config(sys.argv[1])
    rank_pagerank(**config.get_pagerank_params())
