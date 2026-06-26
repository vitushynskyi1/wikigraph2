import struct
import numpy as np
import sys
from translate_output import translate_output
from config import get_params 
from config import change_config 

if len(sys.argv) > 1:
    change_config(sys.argv[1])

window_size, path_compressed_graph, iterations, damping = get_params(
"window_size", "path_compressed_graph", "iterations", "damping")


def iter_decode_rows(payload_bytes, N, window_size=window_size):
    """
    Generator that yields fully reconstructed rows sequentially.
    Manages its own sliding window cache internally.
    """
    window_cache = []
    bp = 0  

    def read_vbyte():
        nonlocal bp
        val = 0
        shift = 0
        while True:
            b = payload_bytes[bp]
            bp += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return val

    for i in range(N):
        flag = payload_bytes[bp]
        bp += 1

        if flag == 0:
            links = []
        elif flag == 255:
            first_val = read_vbyte()
            delta_count = read_vbyte()
            links = [first_val]
            curr = first_val
            for _ in range(delta_count):
                curr += read_vbyte()
                links.append(curr)
        else:
            prev_links = window_cache[-flag]
            links_set = set(prev_links)

            #Perform removes
            remove_count = read_vbyte()
            if remove_count > 0:
                remove_curr = read_vbyte()
                links_set.discard(remove_curr)
                for _ in range(remove_count - 1):
                    remove_curr += read_vbyte()
                    links_set.discard(remove_curr)

            # Perform adds
            add_count = read_vbyte()
            if add_count > 0:
                add_curr = read_vbyte()
                links_set.add(add_curr)
                for _ in range(add_count - 1):
                    add_curr += read_vbyte()
                    links_set.add(add_curr)

            links = sorted(list(links_set))

        yield i, links

        window_cache.append(links)
        if len(window_cache) > window_size:
            window_cache.pop(0)

def rank_pagerank(path_compressed_graph=path_compressed_graph, 
                  iterations=iterations, 
                  damping=damping,
                  window_size=window_size):
    
    print("Opening binary graph stream...")
    with open(path_compressed_graph, "rb") as f:
        N = struct.unpack("<I", f.read(4))[0]
        out_degrees = np.frombuffer(f.read(N * 4), dtype=np.float32)
        payload_bytes = f.read()

    pr_current = np.ones(N, dtype=np.float32) / N
    pr_next = np.zeros(N, dtype=np.float32)

    print(f"Beginning {iterations} iterations of PageRank...")
    for it in range(iterations):
        pr_next.fill(0.0)
        
        for i, in_links in iter_decode_rows(payload_bytes, N, window_size):
            score = 0.0
            for j in in_links:
                score += pr_current[j] / out_degrees[j]
            pr_next[i] = score

        dangling_sum = np.sum(pr_current[out_degrees == 1.0]) 
        base_score = (1.0 - damping) / N + damping * (dangling_sum / N)
        
        pr_current = base_score + (damping * pr_next)
        
        print(f"Iteration {it + 1}/{iterations} completed.")

    print("\nPageRank calculation finished!")
    
    translate_output(pr_current)

if __name__ == "__main__":
    rank_pagerank()
