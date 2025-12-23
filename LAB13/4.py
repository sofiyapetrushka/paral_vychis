import cProfile
from cg_base import cg_base

if __name__ == "__main__":
    cProfile.run("cg_base()", "cg_profile.out")