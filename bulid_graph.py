import re
import scipy.sparse as sp

LINKS_FILE = "dumps/links"
LINKS_PATTERN = re.compile(r"\((\d+),0,(\d+)\)")

def stream_links(filepath, pattern):
    with open(filepath, "r") as f:
        insert_lines = filter(lambda line: line.startswith("INSERT INTO"), f)
        yield from (
            (int(m.group(1)), int(m.group(2)))
            for line in insert_lines
            for m in pattern.finditer(line)
        )

def get_matrix():
    print("Identifying nodes...")
    unique_nodes = {
        node_id 
        for edge in stream_links(LINKS_FILE, LINKS_PATTERN) 
        for node_id in edge
    }
    
    node_to_idx = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    N = len(node_to_idx)
    
    print(f"Graph contains {N} total nodes. Pass 2: Building edges...")
    edges = [
        (node_to_idx[src], node_to_idx[dst])
        for src, dst in stream_links(LINKS_FILE, LINKS_PATTERN)
    ]
    
    rows, cols = zip(*edges) if edges else ([], [])
    data = [1] * len(rows)
    
    adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    print(f"Matrix successfully built! Shape: {adj_matrix.shape} with {adj_matrix.nnz} edges.")
    
    return adj_matrix, node_to_idx

if __name__ == "__main__":
    matrix = get_matrix()
