# talkGPT

Talk to GPT, and GPT responds with a very lifelike voice. Uses GPT output streaming Elevenlabs input + output streaming
for fast response time.

## Installation

1. Make python virtual environment and install python dependencies

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Try running `python3 main.py`. There may be some additional audio dependencies. `brew install` them.

3. Create a .env file, and put in your ElevenLabs and OpenAI API keys. See .env.example.

## TODO

- clean dependencies

- optimize whisper streaming thru non-local model (api?)