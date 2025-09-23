# Parakeet

Parakeet là một dự án nghiên cứu và triển khai **Automatic Speech Recognition (ASR)** với nhiều cách tiếp cận và runtime khác nhau.  
Mục tiêu là đánh giá hiệu năng, độ chính xác và khả năng triển khai của các mô hình ASR trong các môi trường khác nhau.  

## Python code

```bash
nemo["asr"]
torch==2.2.2
torchaudio==2.2.2
transformers==4.44.2
librosa==0.10.2.post1
soundfile==0.12.1
```

- Cài đặt bằng Python thuần với **PyTorch** hoặc **Transformers**.  
- Pipeline bao gồm:  
  1. Tiền xử lý âm thanh  
  2. Trích xuất đặc trưng   
  3. Mô hình hóa 
  4. Decode để ra transcript.  

Ví dụ chạy:

```bash
python parakeet.py
```

## Onnx runtime
```bash
onnx==1.16.2
onnxruntime==1.19.2
onnxruntime-gpu==1.19.2     # nếu chạy GPU, có thể bỏ nếu chỉ dùng CPU
onnxruntime-tools==1.9.0    # optional: tối ưu graph
numpy==1.26.4
```

Mô hình được export sang ONNX để tăng tốc inference trên CPU/GPU.

Dùng ONNX Runtime để benchmark hiệu năng.

Ưu điểm:
- Triển khai cross-platform.
- Hỗ trợ tối ưu hóa graph.

Chạy thử:
```bash
python export_model_parakeet.py
```

## TensorRT runtime
Tích hợp TensorRT để tăng tốc độ suy luận trên GPU NVIDIA.

Dùng ONNX làm input, sau đó build TensorRT engine (.plan).

Benchmark throughput và latency trong môi trường real-time.

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/jovyan/datnt/ASR/encoder-model.onnx \
  --minShapes=audio_signal:1x128x1000,length:1 \
  --optShapes=audio_signal:1x128x12000,length:1 \
  --maxShapes=audio_signal:1x128x20000,length:1 \
  --saveEngine=/home/jovyan/datnt/ASR/encoder-model.plan 
```

