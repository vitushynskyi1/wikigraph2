import pickle

def translate_output(path_compressed_graph, path_pages_dump, winners_listed, rank=[]):
    with open(path_compressed_graph, "rb") as f:
        data = pickle.load(f)
        new_to_old = data["new_to_old"]
        idx_to_node = data["idx_to_node"]
        with open(path_pages_dump, "rw") as pages:
