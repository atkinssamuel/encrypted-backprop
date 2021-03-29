from src.preprocessing import preprocess
from plaintext.plaintext_main import run_plaintext
from benchmark.benchmark_main import run_benchmark

if __name__ == "__main__":
    preprocess()
    run_benchmark()
    # run_plaintext()
    print("Hello World")