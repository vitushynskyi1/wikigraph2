import pickle
import sys
import numpy as np
import re
from config import change_config
from config import get_params

if len(sys.argv) > 1:
    change_config(sys.argv[1])

path_pages, path_compressed_graph, winners_displayed = get_params(
    "path_pages", "path_compressed_graph", "winners_displayed")

def get_titles(ids):
    id_set = set(ids)
    id_to_title = {}
    page_pattern = re.compile(rb"\((\d+),0,'((?:[^'\\]|\\.)*)',")
    with open(path_pages, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO"): 
                continue
            for match in page_pattern.finditer(line):
                page_id = int(match.group(1))
                if page_id in id_set:
                    id_to_title[page_id] = match.group(2)
                    
def translate_output(rank,
                     path_compressed_graph = path_compressed_graph,
                     winners_displayed = winners_displayed):
    with open(path_compressed_graph, "rb") as f:
        data = pickle.load(f)
        new_to_old = data["new_to_old"]
        idx_to_node = data["idx_to_node"]
        winners = np.argsort(rank)[::-1][:winners_displayed]
        winning_page_ids = [idx_to_node[new_to_old[w]] for w in winners]
        titles = get_titles(winning_page_ids)
        
        print(f"\n--- TOP {winners_displayed} WIKIPEDIA PAGES ---")
        for pos, internal_node_idx in enumerate(winners):
            title_str = titles[pos].decode('utf-8', errors='ignore')
            score = rank[internal_node_idx]
            print(f"#{pos + 1}: {title_str} (Score: {score:.6f})")
