import pickle
import sys
import numpy as np
import re
from config import get_params
from config import change_config

if len(sys.argv) > 1:
    change_config(sys.argv[1])

path_pages, path_metadata, winners_displayed  = get_params("path_pages", "path_metadata", "winners_displayed")

def get_titles(ids, path_pages = path_pages):
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
    return id_to_title

def translate_output(rank, 
                     path_metadata=path_metadata, 
                     winners_count=winners_displayed):
    
    print("Loading graph metadata dictionaries...")
    with open(path_metadata, "rb") as f:
        data = pickle.load(f)
        new_to_old = data["new_to_old"]
        idx_to_node = data["idx_to_node"]
        
    winners = np.argsort(rank)[::-1][:winners_count]
    
    winning_page_ids = [idx_to_node[new_to_old[w]] for w in winners]
    
    print("Scanning SQL database for article titles...")
    titles_dict = get_titles(winning_page_ids)
        
    print(f"\n--- TOP {winners_count} WIKIPEDIA PAGES ---")
    for pos, internal_node_idx in enumerate(winners):
        real_page_id = idx_to_node[new_to_old[internal_node_idx]]
        
        title_bytes = titles_dict.get(real_page_id, b"<TITLE NOT FOUND>")
        title_str = title_bytes.decode('utf-8', errors='ignore').replace("\\'", "'")
        
        score = rank[internal_node_idx]
        
        print(f"#{pos + 1}: {title_str} (ID: {real_page_id}) (Score: {score:.6f})")
