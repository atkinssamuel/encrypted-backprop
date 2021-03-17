class Directories:
    """
    Contains all of the directories for this repository
    """
    raw = "raw/"
    data = "data/"
    src = "src/"
    benchmark = "benchmark/"
    benchmark_models = benchmark + "models/"
    benchmark_other_models = benchmark_models + "other/"
    benchmark_best_model = benchmark_models + "best/"
    benchmark_results = benchmark + "results/"

    # plaintext directories
    plaintext = "plaintext/"
    plaintext_results = plaintext + "results/"


class Files:
    """
    Contains all of the file names for this repository
    """
    # data files
    zipped_data = Directories.raw + "creditcard.zip"
    raw_data = Directories.raw + "creditcard.csv"
    balanced_data = Directories.data + "balanced.pkl"

    # benchmark files
    benchmark_training_loss = Directories.benchmark_results + "training_loss.png"
    benchmark_training_accuracy = Directories.benchmark_results + "training_accuracy.png"
    benchmark_validation_loss = Directories.benchmark_results + "validation_loss.png"
    benchmark_validation_accuracy = Directories.benchmark_results + "validation_accuracy.png"

    # plaintext files
    plaintext_training_loss = Directories.plaintext_results + "training_loss.png"
    plaintext_training_accuracy = Directories.plaintext_results + "training_accuracy.png"
    plaintext_validation_loss = Directories.plaintext_results + "validation_loss.png"
    plaintext_validation_accuracy = Directories.plaintext_results + "validation_accuracy.png"


class Benchmark:
    """
    Contains the hyper-parameters for the benchmark network
    """
    epoch_count = 50
    batch_size = 1024*2
    learning_rate = 0.001
    checkpoint_frequency = 20


class Plaintext:
    """
    Contains the hyper-parameters for the plaintext network
    """
    epoch_count = 50
    batch_size = 1024*2
    learning_rate = 0.001
    checkpoint_frequency = 20