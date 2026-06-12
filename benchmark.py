import time
import numpy as np
import onnxruntime as ort


def benchmark_model(model_path, input_shape, provider="CPUExecutionProvider", num_iterations=100, warmup=10):
    """Benchmark an ONNX model on a given execution provider.
    
    Args:
        model_path: Path to .onnx model file
        input_shape: Tuple of input dimensions (batch, channels, height, width)
        provider: ONNX Runtime execution provider
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with latency statistics
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=[provider])
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    return {
        "provider": provider,
        "input_shape": input_shape,
        "iterations": num_iterations,
        "avg_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
    }


def print_results(results):
    """Pretty-print benchmark results."""
    print(f"\n{'='*50}")
    print(f"Provider: {results['provider']}")
    print(f"Input shape: {results['input_shape']}")
    print(f"Iterations: {results['iterations']}")
    print(f"{'='*50}")
    print(f"  Avg latency:   {results['avg_ms']:.3f} ms")
    print(f"  P50 latency:   {results['p50_ms']:.3f} ms")
    print(f"  P95 latency:   {results['p95_ms']:.3f} ms")
    print(f"  P99 latency:   {results['p99_ms']:.3f} ms")
    print(f"  Throughput:    {results['throughput_fps']:.1f} FPS")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ONNX Inference Benchmark")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    
    input_shape = (args.batch_size, 3, 224, 224)
    results = benchmark_model(args.model, input_shape, args.provider, args.iterations)
    print_results(results)
