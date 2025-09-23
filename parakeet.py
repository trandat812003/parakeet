import nemo.collections.asr as nemo_asr

# Load model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# Audio path
audio_path = '/media/admin123/DataVoice/aaa/demo_1p.wav'

# Transcribe without timestamps (just plain text)


# Transcribe with timestamps
output = asr_model.transcribe([audio_path], timestamps=True)
print("Transcript:", output[0].text)
# Get timestamps
segment_timestamps = output[0].timestamp.get('segment', [])
word_timestamps = output[0].timestamp.get('word', [])
char_timestamps = output[0].timestamp.get('char', [])

# Print segment-level timestamps
print("\n[Segment-level timestamps]")
for stamp in segment_timestamps:
    print(f"{stamp['start']}s - {stamp['end']}s : {stamp['segment']}")

# # Optional: print other levels
# print("\n[Word-level timestamps]")
# for stamp in word_timestamps:
#     print(f"{stamp['start']}s - {stamp['end']}s : {stamp['word']}")

# print("\n[Char-level timestamps]")
# for stamp in char_timestamps:
#     print(f"{stamp['start']}s - {stamp['end']}s : {stamp['char']}")