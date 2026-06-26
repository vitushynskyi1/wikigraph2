import pickle
import numpy as np
import sys
import config

def iter_decode_rows(compressed_rows, window_size=7):
    """
    Generator that yields fully reconstructed rows sequentially.
    Manages its own sliding window cache internally.
    """
    window_cache = []

    for i, row_data in enumerate(compressed_rows):
        ref_offset = row_data.get("ref_offset", 0)

        if ref_offset > 0:
            prev_links = window_cache[-ref_offset]
            links_set = set(prev_links)

            #remove instructions
            for r in row_data.get("remove", []):
                links_set.discard(r)

            #add instructions
            add_first = row_data.get("add_first", -1)
            if add_first != -1:
                links_set.add(add_first)
                curr = add_first
                for d in row_data.get("add_deltas", []):
                    curr += d
                    links_set.add(curr)

            links = sorted(list(links_set))

        else: #reference encoding wasn't applied
            first = row_data.get("first", -1)
            if first == -1:
                links = []
            else:
                deltas = row_data.get("deltas", [])
                links = [first]
                curr = first
                for d in deltas:
                    curr += d
                    links.append(curr)

        yield i, links

        #sliding window cache update
        window_cache.append(links)
        if len(window_cache) > window_size:
            window_cache.pop(0)


def rank_pagerank(path_compressed_graph, iterations, damping):
    print("Loading graph metadata...")
    with open(path_compressed_graph, "rb") as f:
        data = pickle.load(f)
        
    N = data["N"]
    out_degrees = data["out_degrees"]
    compressed_rows = data["compressed_rows"]
    new_to_old = data["new_to_old"]
    idx_to_node = data["idx_to_node"]

    pr_current = np.ones(N) / N
    pr_next = np.zeros(N)
    
    # We can fetch the window_size from the config system if needed, 
    # but defaulting to 7 covers our basic.toml configuration natively.
    window_size = config.get_compress_params().get("window_size", 7)

    print(f"Beginning {iterations} iterations of Streaming PageRank...")
    for it in range(iterations):
        pr_next.fill(0.0)
        
        for i, in_links in iter_decode_rows(compressed_rows, window_size):
            
            score = 0.0
            for j in in_links:
                score += pr_current[j] / out_degrees[j]
                
            pr_next[i] = score

        dangling_sum = np.sum(pr_current[out_degrees == 1.0]) 
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.change_config(sys.argv[1])
    rank_pagerank(**config.get_pagerank_params())
