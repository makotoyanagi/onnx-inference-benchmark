# ONNX Inference Benchmark

Benchmarking ONNX Runtime inference across hardware backends.

## Backends
- CPU (default)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- TensorRT (NVIDIA optimized)

## Models
- ResNet-50 (image classification)
- BERT-base (text classification)
- YOLOv5 (object detection)

## Usage
```bash
python benchmark.py --model resnet50 --providers cpu cuda rocm
```
