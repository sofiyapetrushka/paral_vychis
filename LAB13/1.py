import pstats

if __name__ == "__main__":
    p = pstats.Stats("cg_profile.out")
    p.strip_dirs().sort_stats("cumtime").print_stats(10)