from src.preprocessing import preprocess
from src.benchmark.benchmark_main import run_benchmark

if __name__ == "__main__":
    preprocess()
    run_benchmark()
    print("Hello World")