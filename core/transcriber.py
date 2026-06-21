import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL",default="small")

_model = None

def load_model():
    global _model
    if _model is None:
        print("="*40)
        print("\n Loading model ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("="*40)
        print("\n Model Loaded.")
    return _model

def transcribe_chunk(chunk_path : str, translate : bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task = task)
    return result['text']
    