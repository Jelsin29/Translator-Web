import gradio as gr
import whisper
from translate import Translator

def translator(audio_file):
   #1. Transcrip text

   # https://github.com/openai/whisper.git
   # Alternetives https://assemblyai.com
   # Whisper (https://whisper.ai)

   try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish")
        transcription = result["text"]
   except Exception as e:
         raise gr.Error(
              f"Se ha producido un error al transcribir el texto: {str(e)}")
   
   #2. Translate text
   # https://pypi.org/project/translate/

   en_transcription = Translator(to_lang="en").translate(transcription)

   # 3. Generate translate audio
   

web = gr.Interface(   
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
    ),
    outputs=[],
    title="Speech to Text Translator",
    description="A speech to text translator that converts spoken words into text."
)

web.launch()
