import whisper
from gtts import gTTS
import soundfile as sf
import gradio as gr
import os
import numpy as np
import noisereduce as nr

# Create static directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Load Whisper model
model = whisper.load_model("small")

def pipeline(audio):
    try:
        # Debug: Inspect audio input
        if audio is None:
            raise ValueError("Audio input is None")
        sample_rate, audio_data = audio
        print(f"Sample rate: {sample_rate}")
        print(f"Audio data shape: {audio_data.shape}")
        print(f"Audio data type: {audio_data.dtype}")

        # Normalize audio data if needed (ensure it's in the correct format)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            print("Audio data normalized to float32")

        # Apply noise reduction
        audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate)
        print("Noise reduction applied")

        # Save input audio to a file
        input_path = "static/input.wav"
        sf.write(input_path, audio_data, sample_rate)
        print(f"Audio saved to: {input_path}")

        # Verify the file was created
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Failed to create {input_path}")

        # Transcribe using Whisper
        result = model.transcribe(input_path)
        transcription = result["text"]
        print(f"Transcription: {transcription}")

        # Simulate a dummy LLM response
        llm_response = f"Hi Apoorv, you said: '{transcription}'. I'm your AI assistant, ready to help!"
        print(f"LLM Response: {llm_response}")

        # Synthesize voice with gTTS
        output_path = "static/output.mp3"  # gTTS saves as MP3
        tts = gTTS(text=llm_response, lang='en')
        tts.save(output_path)
        print(f"Output audio saved to: {output_path}")

        return transcription, llm_response, output_path
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise e

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Real-Time Whisper + gTTS Demo")
    audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your voice")
    transcription_output = gr.Textbox(label="Transcription")
    response_output = gr.Textbox(label="LLM Response")
    audio_output = gr.Audio(label="Synthesized Response")
    btn = gr.Button("Process")
    btn.click(fn=pipeline, inputs=audio_input, outputs=[transcription_output, response_output, audio_output])

if __name__ == "__main__":
    demo.launch(debug=True)