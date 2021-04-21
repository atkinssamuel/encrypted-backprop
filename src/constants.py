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
    # androgen data file
    androgen_data = Directories.raw + "qsar_androgen_receptor.csv"

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
    epoch_count = 1
    batch_size = 64
    learning_rate = 0.005
    checkpoint_frequency = 1


class Plaintext:
    """
    Contains the hyper-parameters for the plaintext network
    """
    epoch_count = 1
    batch_size = 64
    learning_rate = 0.005
    checkpoint_frequency = 1
