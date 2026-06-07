import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

class VoiceIdentityEngine:
    def __init__(self):
        print("🧠 Loading ECAPA-TDNN Model (This will download ~80MB the first time)...")
        # Downloads and caches the model locally
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"} # Forcing CPU to keep it cheap and serverless-friendly
        )
        print("✅ Model Loaded Successfully.\n")

    def generate_embedding(self, audio_file_path):
        """
        Takes a clean 16kHz WAV file and returns a 192-dimensional numerical array.
        """
        # Load the audio file into a PyTorch tensor
        signal, fs = torchaudio.load(audio_file_path)
        
        # Pass it through the neural network
        embeddings = self.model.encode_batch(signal)
        
        # Squeeze the tensor and convert it to a standard Python list for JSON export
        vector = embeddings.squeeze().tolist()
        return vector