import os
import time
import asyncio
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from tritonclient.http.aio import InferenceServerClient
from tritonclient.http import InferInput, InferRequestedOutput

# Cấu hình
TRITON_URL = "4d78b93b9eae.ngrok-free.app"
COMBINED_MODEL = "combined"
VAD_DIR = "/media/admin123/DataVoice/aaa"
SAMPLE_RATE = 16000

processor = AudioToMelSpectrogramPreprocessor(
    sample_rate=SAMPLE_RATE,
    normalize="per_feature",
    window_size=0.025,
    window_stride=0.01,
    window="hann",
    features=128,
    n_fft=512,
    log=True,
    frame_splicing=1,
    dither=1e-05,
    pad_to=0,
    pad_value=0.0,
)


def construct_states(batch: int):
    """ Trả về các mảng numpy cho các state tensors cần thiết """
    targets = np.zeros((batch, 1), dtype=np.int32)
    target_length = np.ones((batch,), dtype=np.int32)
    s1 = np.zeros((2, batch, 640), dtype=np.float32)
    s2 = np.zeros((2, 1, 640), dtype=np.float32)
    return targets, target_length, s1, s2

async def process_file(client: InferenceServerClient, wav_filename: str):
    wav_path = os.path.join(VAD_DIR, wav_filename)
    audio, sr = sf.read(wav_path)
    assert sr == SAMPLE_RATE, f"sample rate mismatch: got {sr}"

    duration_seconds = audio.shape[0] / SAMPLE_RATE
    batch = 1

    # Preprocess thành mel-spectrogram
    audio_sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    length_tensor = torch.tensor([audio_sig.shape[1]], dtype=torch.int64)

    processed_signal, processed_length = processor(input_signal=audio_sig, length=length_tensor)
    audio_signal_np = processed_signal.numpy()
    length_np = processed_length.numpy()
    breakpoint()

    # Chuẩn bị inputs
    inp_signal = InferInput("audio_signal", audio_signal_np.shape, "FP32")
    inp_length = InferInput("length", length_np.shape, "INT64")

    inp_signal.set_data_from_numpy(audio_signal_np)
    inp_length.set_data_from_numpy(length_np)

    # States & targets
    targets, target_length, s1, s2 = construct_states(batch)
    inp_targets = InferInput("targets", targets.shape, "INT32")
    inp_tlen = InferInput("target_length", target_length.shape, "INT32")
    inp_s1 = InferInput("states.1", s1.shape, "FP32")
    inp_s2 = InferInput("onnx::Slice_3", s2.shape, "FP32")

    inp_targets.set_data_from_numpy(targets)
    inp_tlen.set_data_from_numpy(target_length)
    inp_s1.set_data_from_numpy(s1)
    inp_s2.set_data_from_numpy(s2)

    start_infer = time.time()
    resp = await client.infer(
        model_name=COMBINED_MODEL,
        inputs=[inp_signal, inp_length, inp_targets, inp_tlen, inp_s1, inp_s2],
        outputs=[InferRequestedOutput("final_outputs", binary_data=True)]
    )
    infer_time = time.time() - start_infer

    out = resp.as_numpy("final_outputs")
    if out is None:
        print("No output received for 'final_outputs'")
        return 0.0

    # Nếu output có dim4, squeeze
    if out.ndim == 4:
        out = np.squeeze(out, axis=2)

    rtf = duration_seconds / infer_time if infer_time > 0 else float("inf")
    print(
        f"{wav_filename} → output shape {out.shape}, "
        f"duration: {duration_seconds:.2f}s, "
        f"infer time: {infer_time:.3f}s, RTFx ≈ {rtf:.3f}×"
    )
    return rtf

async def main():
    client = InferenceServerClient(url=TRITON_URL, ssl=True, conn_limit=10, conn_timeout=1800)
    try:
        files = [f for f in sorted(os.listdir(VAD_DIR)) if f.lower().endswith(".wav")]
        total_rtf = 0.0

        start_all = time.time()
        for fname in files:
            rtf = await process_file(client, fname)
            total_rtf += rtf
        total_time = time.time() - start_all
        avg_rtf = total_rtf / len(files) if files else 0.0

        print(f"\nProcessed {len(files)} files in {total_time:.2f}s, Average RTFx ≈ {avg_rtf:.3f}×")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
