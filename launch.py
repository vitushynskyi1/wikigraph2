import build_graph
import compress_graph
import rank_pagerank
import sys
import config
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config.change_config(sys.argv[1])
    build_graph.build_graph()
    compress_graph.compress_graph(**config.get_compress_params())
    rank_pagerank.rank_pagerank(**config.get_pagerank_params())
