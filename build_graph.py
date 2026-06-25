import re
import pickle
import scipy.sparse as sp

PAGES_FILE = "dumps/pages"
LT_FILE = "dumps/linktargets"
LINKS_FILE = "dumps/links"

PAGE_PATTERN = re.compile(rb"\((\d+),0,'((?:[^'\\]|\\.)*)',")
LT_PATTERN = re.compile(rb"\((\d+),0,'((?:[^'\\]|\\.)*)'\)")
LINKS_PATTERN = re.compile(rb"\((\d+),0,(\d+)\)")

def stream_insert_lines(filepath):
    with open(filepath, "rb") as f:
        yield from filter(lambda line: line.startswith(b"INSERT INTO"), f)

def get_matrix():
    print("Processing pages...")
    title_to_page_id = {}
    for line in stream_insert_lines(PAGES_FILE):
        for match in PAGE_PATTERN.finditer(line):
            title_to_page_id[match.group(2)] = int(match.group(1))

    node_to_idx = {page_id: idx for idx, page_id in enumerate(title_to_page_id.values())}
    N = len(node_to_idx)
    print(f"Graph contains exactly {N} valid articles.")

    print("Processing linktargets...")
    lt_to_page_id = {}
    for line in stream_insert_lines(LT_FILE):
        for match in LT_PATTERN.finditer(line):
            title_bytes = match.group(2)
            if title_bytes in title_to_page_id:
                lt_id = int(match.group(1))
                lt_to_page_id[lt_id] = title_to_page_id[title_bytes]

    print("Processing edges...")
    rows = []
    cols = []
    
    for line in stream_insert_lines(LINKS_FILE):
        for match in LINKS_PATTERN.finditer(line):
            from_page_id = int(match.group(1))
            to_lt_id = int(match.group(2))
            
            if from_page_id in node_to_idx and to_lt_id in lt_to_page_id:
                target_page_id = lt_to_page_id[to_lt_id]
                
                rows.append(node_to_idx[from_page_id])
                cols.append(node_to_idx[target_page_id])

    data = [1] * len(rows)
    adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    print(f"Matrix successfully built! Shape: {adj_matrix.shape} with {adj_matrix.nnz} edges.")
    
    return adj_matrix, node_to_idx

def build_graph():
    matrix, mapping = get_matrix()
    
    print("Saving raw graph to disk...")
    with open("raw_graph.pkl", "wb") as f:
        pickle.dump({
            "matrix": matrix,
            "node_to_idx": mapping
        }, f)
    print("Raw graph saved as raw_graph.pkl!")

if __name__ == "__main__":
    build_graph()
