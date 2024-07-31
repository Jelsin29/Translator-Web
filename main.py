import gradio as gr

def translator():
    pass

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
