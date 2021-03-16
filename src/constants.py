class Directories:
    raw = "raw/"
    data = "data/"
    src = "src/"
    benchmark = src + "benchmark/"
    benchmark_models = benchmark + "models/"
    benchmark_other_models = benchmark_models + "other/"
    benchmark_best_model = benchmark_models + "best/"
    benchmark_results = benchmark + "results/"


class Files:
    zipped_data = Directories.raw + "creditcard.zip"
    raw_data = Directories.raw + "creditcard.csv"
    balanced_data = Directories.data + "balanced.pkl"
    benchmark_training_loss = Directories.benchmark_results + "training_loss.png"
    benchmark_training_accuracy = Directories.benchmark_results + "training_accuracy.png"
    benchmark_validation_loss = Directories.benchmark_results + "validation_loss.png"
    benchmark_validation_accuracy = Directories.benchmark_results + "validation_accuracy.png"


class Benchmark:
    epoch_count = 50
    batch_size = 1024*2
    learning_rate = 0.001
    shuffle_flag = True
    checkpoint_frequency = 20
