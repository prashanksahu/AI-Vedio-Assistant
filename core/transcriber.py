import whisper
import os
from rich import print

WHISPER_MODEL = os.getenv("WHISPER_MODEL",default="small")

_model = None

#Downlaoding and loading the whisper model locally
def load_model():
    global _model
    if _model is None:
        print("\n Loading model ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("\n Model Loaded.")
    return _model

#Transcribing a single chunk
def transcribe_chunk(chunk_path : str, translate : bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task = task)
    return result['text']
    
#Transcribing all the chunks at once
def transcribe_all(chunks : list, translate : bool = False) -> str:
    full_transcript = ""
    for i,chunk in enumerate(chunks):
        print(f"\nTranscribing chunk.. {i+1}")
        text = transcribe_chunk(chunk,translate=translate)
        full_transcript += text + " "
    print("\nTranscription completed")
    return full_transcript

