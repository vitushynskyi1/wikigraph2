import build_graph
import compress_graph
import rank_pagerank
import sys
from config import change_config
if __name__ == "__main__":
    if len(sys.argv) > 1:
        change_config(sys.argv[1])
    build_graph.build_graph()
    compress_graph.compress_graph()
    rank_pagerank.rank_pagerank()
