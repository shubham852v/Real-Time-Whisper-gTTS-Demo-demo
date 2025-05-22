# 1. Clone the repository
https://github.com/shubham852v/Real-Time-Whisper-gTTS-Demo-demo
cd real-time-avatar-demo

# 2. Create and activate a virtual environment (venv)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. install the required packages in the new virtual environment:
pip install whisper gradio noisereduce numpy soundfile gTTS

# 5. Run the app:
python app.py
