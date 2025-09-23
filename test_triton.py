import os, time, asyncio, numpy as np, soundfile as sf
import tritonclient.http.aio as aioclient
from tritonclient.http import InferInput, InferRequestedOutput

TRITON_URL = "https://4d78b93b9eae.ngrok-free.app"
COMBINED_MODEL = "combined"
VAD_DIR = "test_dataa"
SAMPLE_RATE = 16000

async def infer_one(client, audio_signal, length, wav):
    # Chuẩn bị inputs
    inputs = []
    inputs.append(InferInput("audio_signal", audio_signal.shape, "FP32"))
    inputs.append(InferInput("length",       length.shape,       "INT64"))
    batch = audio_signal.shape[0]
    targets = np.zeros((batch,1), dtype=np.int32)
    target_length = np.ones((batch,), dtype=np.int32)
    s1 = np.zeros((2,batch,640), dtype=np.float32)
    s2 = np.zeros((2,1,640), dtype=np.float32)
    inputs.append(InferInput("targets",       targets.shape,       "INT32"))
    inputs.append(InferInput("target_length", target_length.shape, "INT32"))
    inputs.append(InferInput("states.1",      s1.shape,            "FP32"))
    inputs.append(InferInput("onnx::Slice_3",  s2.shape,            "FP32"))
    for inp, arr in zip(inputs, [audio_signal, length, targets, target_length, s1, s2]):
        inp.set_data_from_numpy(arr)

    start_inf = time.time()
    result = await client.infer(
        COMBINED_MODEL,
        inputs=inputs,
        outputs=[InferRequestedOutput("final_outputs")]
    )
    infer_time = time.time() - start_inf
    out = result.as_numpy("final_outputs")
    audio_secs = audio_signal.shape[2] / SAMPLE_RATE
    rtf = infer_time / audio_secs if audio_secs > 0 else float('inf')
    print(f"{wav}: shape={out.shape}, duration={audio_secs:.2f}s, infer={infer_time:.3f}s, RTFx={rtf:.3f}")
    return audio_secs


async def main():
    files = [f for f in os.listdir(VAD_DIR) if f.endswith('.wav')]
    total_audio = 0.0
    async with aioclient.InferenceServerClient(url=TRITON_URL, ssl=True, conn_limit=10) as client:
        tasks = []
        for wav in files:
            audio, sr = sf.read(os.path.join(VAD_DIR, wav), dtype='float32')
            assert sr == SAMPLE_RATE
            audio = np.expand_dims(audio, 0).astype('float32')
            length = np.array([audio.shape[2]], dtype=np.int64)
            total_audio += audio.shape[2] / SAMPLE_RATE
            task = asyncio.create_task(infer_one(client, audio, length, wav))
            tasks.append(task)
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"Finished {len(files)} files in {elapsed:.2f}s → throughput {total_audio/elapsed:.2f} audio‑s/s")

if __name__=="__main__":
    asyncio.run(main())
