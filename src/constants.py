class Directories:
    raw = "raw/"
    data = "data/"
    src = "src/"
    benchmark = src + "benchmark/"
    benchmark_models = benchmark + "models/" \
                                   ""

class Files:
    zipped_data = Directories.raw + "creditcard.zip"
    raw_data = Directories.raw + "creditcard.csv"
    balanced_data = Directories.data + "balanced.pkl"
