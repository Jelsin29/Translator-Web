import torch
import whisper
import gradio as gr
from translate import Translator
from dotenv import dotenv_values
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

config = dotenv_values(".env")

ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

def translator(audio_file):
    #1. Transcript text

    # https://github.com/openai/whisper.git
    # Alternatives https://assemblyai.com
    # Whisper (https://whisper.ai)

    try:
        model = whisper.load_model("base", device=torch.device("cpu"))
        result = model.transcribe(audio_file, language="Spanish", fp16=False) 
        transcription = result["text"]
    except Exception as e: 
        raise gr.Error(
                f"Se ha producido un error al transcribir el texto: {str(e)}")
    print(f"Texto orginal: {transcription}")

    #2. Translate text
    # https://pypi.org/project/translate/

    try:     
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error al traducir el texto: {str(e)}")
    print(f"Texto traducido: {en_transcription}")



    # 3. Generate translate audio

    # https://elevenlabs.io/docs/api-reference/getting-started

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",  # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = "audios/en.mp3"

        with open(save_file_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)
    except FileNotFoundError as e:
        raise gr.Error( f"Se ha producido un error al crear el audio: {str(e)}")


    return save_file_path


web = gr.Interface(   
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[gr.Audio(label="Ingles")],
    title="Speech to Text Translator",
    description="A speech to text translator that converts spoken words into text."
)

web.launch()
