# 🦜 parakeet-fastAPI
URL: [https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)

## 📊 Survey

### 🔹 Batching
- `max_batch = 4`
- Trong một khoảng thời gian `batch_ms`, hệ thống sẽ gom các request lại để xử lý.
- Nếu đủ batch → chạy inference, nếu chưa đủ thì chờ trong `batch_ms`.
- Nếu gửi đoạn audio ngắn hơn `32ms` → **không có kết quả**.

### 🔹 Luồng xử lý model
- Hệ thống gồm 2 phần:
  1. **Phần chính**: nhận request từ FastAPI.
  2. **Phần phụ**: gom batch và xử lý inference.
- Quy trình:
  - FastAPI nhận request.
  - Gom request thành batch theo `batch_ms`.
  - Chạy inference.
  - Trả kết quả cho client.

### 🔹 VAD (Voice Activity Detection)
- Chia audio thành các **chunk** theo tín hiệu giọng nói.
- Nguyên tắc:
  - Khi VAD phát hiện tín hiệu bắt đầu → đến khi phát hiện tín hiệu kết thúc.
  - `min_silence_duration_ms = 30ms`
  - `speech_pad_ms = 120ms`

### 🔹 Timestamp & Alignment
- Nếu **không cần timestamp** → inference nhanh hơn.
- Nếu **cần timestamp** → bật chế độ alignment để căn chỉnh transcript theo thời gian.
- **Alignment**:
  - Dùng để xác định từ bắt đầu/kết thúc tại thời điểm nào trong audio.
  - Khi bật → inference chậm hơn.
  - Khi tắt (comment đoạn code alignment) → inference nhanh hơn.

---

## 🚀 Tóm tắt
- FastAPI nhận request → gom batch trong `batch_ms` → chạy inference → trả kết quả.
- VAD cắt audio thành các đoạn có tiếng nói.
- Timestamp chỉ có khi bật alignment (đổi lại tốc độ sẽ chậm hơn).

## 📈 Kết quả thử nghiệm

| Config                        | 1p                          | 3p                          | 15p                         |
|-------------------------------|-----------------------------|-----------------------------|-----------------------------|
| **FastAPI (batching)**        |         min-max(AVG) <br> GPU util <br> memory       |                             |                             |
| With chunk, With timestamp    | 3.736-11.469 (6.439) <br> 20% <br> 3059MB | 4.540-18.516(10200) <br> 41% <br> 3347MB | 24.838-50.449 (41.633 ms) <br> 50% <br> 41633MB |
| With chunk, Without timestamp | 0.589-3.718(1.410) <br> 30% <br> 3229MB | 0.764-6.926(3.624) <br> 43% <br> 3515MB | 11.709-22.519(17.835) <br> 50% <br> 3545MB |
| Without chunk, With timestamp |  <br> % <br> MB |  <br> % <br> MB |  <br> % <br> MB |
| Without chunk, Without timestamp |  0.071-0.128(0.121)<br> 43% <br> 3591MB | 0.101-0.360(0.308)ms<br> 66% <br> 3791MB | 1.664-2.957(2.774)<br> 80% <br> 23057MB |
| **Streaming** with 1024byte           |           3.243 <br> 5% <br> 2700MB           |    8.855 <br> 5% <br> 2700MB      |  34.824 <br> 5% <br> 2700MB   |
