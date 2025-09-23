import os
import torch
import nemo.collections.asr as nemo_asr
import onnx
import json

def export_parakeet_components():
    # 1. Load model
    print("📦 Đang load model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    model.eval()
    model.to('cpu')
    
    # 2. Tạo export directory
    export_dir = "parakeet_tdt_export"
    os.makedirs(export_dir, exist_ok=True)
    
    # 3. Export Encoder
    print("🔤 Đang export Encoder...")
    
    # Tạo wrapper class để handle kwargs
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            
        def forward(self, audio_signal, length):
            return self.encoder(audio_signal=audio_signal, length=length)
    
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()
    
    # Tạo dummy input cho encoder (audio features)
    dummy_audio_signal = torch.randn(1, 16000)  # 1 giây audio 16kHz
    dummy_audio_length = torch.tensor([16000])
    
    # Get processed features từ preprocessor
    with torch.no_grad():
        processed_signal, processed_length = model.preprocessor(
            input_signal=dummy_audio_signal,
            length=dummy_audio_length
        )
    
    encoder_output_path = os.path.join(export_dir, "parakeet_encoder.onnx")
    torch.onnx.export(
        encoder_wrapper,
        (processed_signal, processed_length),
        encoder_output_path,
        input_names=['audio_features', 'audio_length'],
        output_names=['encoder_outputs', 'encoder_length'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 2: 'time_steps'},
            'audio_length': {0: 'batch_size'},
            'encoder_outputs': {0: 'batch_size', 1: 'time_steps'},
            'encoder_length': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=True
    )
    
    # 4. Export Decoder/Prediction Network
    print("🧠 Đang export Decoder...")
    
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            
        def forward(self, targets, target_length):
            outputs = self.decoder(targets=targets, target_length=target_length)
            # Nếu output là tuple, chỉ return tensor đầu tiên cho ONNX export
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs
    
    decoder_wrapper = DecoderWrapper(model.decoder)
    decoder_wrapper.eval()
    
    # Dummy input cho decoder (previous tokens)
    dummy_targets = torch.randint(0, model.decoder.vocab_size, (1, 10))  # batch_size=1, seq_len=10
    dummy_target_length = torch.tensor([10])
    
    decoder_output_path = os.path.join(export_dir, "parakeet_decoder.onnx")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_targets, dummy_target_length),
        decoder_output_path,
        input_names=['targets', 'target_length'],
        output_names=['decoder_outputs'],
        dynamic_axes={
            'targets': {0: 'batch_size', 1: 'target_length'},
            'target_length': {0: 'batch_size'},
            'decoder_outputs': {0: 'batch_size', 1: 'target_length'}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=True
    )
    
    # 5. Export Joint Network
    print("🔗 Đang export Joint Network...")
    
    class JointWrapper(torch.nn.Module):
        def __init__(self, joint):
            super().__init__()
            self.joint = joint
            
        def forward(self, encoder_outputs, decoder_outputs):
            # Chỉ sử dụng joint network để compute logits, không compute loss
            # Bypass loss computation bằng cách không pass transcripts
            joint_net = self.joint.joint_net if hasattr(self.joint, 'joint_net') else self.joint
            
            # Direct computation without loss
            if hasattr(joint_net, 'joint'):
                return joint_net.joint(encoder_outputs, decoder_outputs)
            else:
                # Fallback: try to access the core joint computation
                return joint_net(encoder_outputs, decoder_outputs)
    
    joint_wrapper = JointWrapper(model.joint)
    joint_wrapper.eval()
    
    # Dummy inputs cho joint network
    # Get actual dimensions từ model
    with torch.no_grad():
        # Test với encoder wrapper để lấy output shape
        enc_out, enc_len = encoder_wrapper(processed_signal, processed_length)
        dec_out = decoder_wrapper(dummy_targets, dummy_target_length)
        
        # Handle decoder output (có thể là tuple hoặc tensor)
        if isinstance(dec_out, tuple):
            dec_tensor = dec_out[0]  # Lấy tensor đầu tiên
            print(f"🔍 Decoder output type: tuple với {len(dec_out)} elements")
        else:
            dec_tensor = dec_out
            print(f"🔍 Decoder output type: tensor")
            
        encoder_dim = enc_out.shape[-1]
        decoder_dim = dec_tensor.shape[-1]
        
        print(f"📏 Encoder dim: {encoder_dim}, Decoder dim: {decoder_dim}")
        print(f"📏 Encoder shape: {enc_out.shape}")
        print(f"📏 Decoder shape: {dec_tensor.shape}")
    
    # Tạo dummy inputs với đúng dimensions  
    dummy_encoder_out = torch.randn(1, 100, encoder_dim)  # [batch, time, enc_dim]
    dummy_decoder_out = torch.randn(1, 10, decoder_dim)   # [batch, targets, dec_dim]
    
    joint_output_path = os.path.join(export_dir, "parakeet_joint.onnx")
    torch.onnx.export(
        joint_wrapper,
        (dummy_encoder_out, dummy_decoder_out),
        joint_output_path,
        input_names=['encoder_outputs', 'decoder_outputs'],
        output_names=['joint_outputs'],
        dynamic_axes={
            'encoder_outputs': {0: 'batch_size', 1: 'time_steps'},
            'decoder_outputs': {0: 'batch_size', 1: 'target_length'},
            'joint_outputs': {0: 'batch_size', 1: 'time_steps', 2: 'target_length'}
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=True
    )
    
    # 6. Verify exported models
    print("\n🔍 Đang verify các file ONNX...")
    for filename in ["parakeet_encoder.onnx", "parakeet_decoder.onnx", "parakeet_joint.onnx"]:
        filepath = os.path.join(export_dir, filename)
        try:
            onnx_model = onnx.load(filepath)
            onnx.checker.check_model(onnx_model)
            print(f"✅ {filename}: OK")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
    
    # 7. Save model config
    print("\n📝 Đang save model config...")
    config_path = os.path.join(export_dir, "model_config.json")
    
    # Lấy config từ model
    try:
        vocab_size = getattr(model.decoder, 'vocab_size', 1024)
        sample_rate = getattr(model.preprocessor.featurizer, 'sample_rate', 16000) if hasattr(model.preprocessor, 'featurizer') else 16000
        n_mels = getattr(model.preprocessor.featurizer, 'n_mels', 80) if hasattr(model.preprocessor, 'featurizer') else 80
    except:
        vocab_size = 1024
        sample_rate = 16000
        n_mels = 80
    
    config_dict = {
        "model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "vocab_size": vocab_size,
        "encoder_dim": encoder_dim,
        "decoder_dim": decoder_dim,
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "tokenizer_type": "bpe",
        "files": {
            "encoder": "parakeet_encoder.onnx",
            "decoder": "parakeet_decoder.onnx", 
            "joint": "parakeet_joint.onnx"
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Export hoàn tất! Các file được lưu trong: {export_dir}")
    print("📁 Files:")
    for file in os.listdir(export_dir):
        filepath = os.path.join(export_dir, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   - {file}: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_parakeet_components()