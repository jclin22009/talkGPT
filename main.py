import os
from openai import OpenAI
import speech_recognition as sr
import elevenlabs
from dotenv import load_dotenv

load_dotenv()
elevenlabs.set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conversation = [
    {
        "role": "system",
        "content": open("prompt.txt", "r").read(),
    }
]
recognizer = sr.Recognizer()
mic = sr.Microphone()

def transcribe_speech(language: str) -> str:
    """
    Records audio from the microphone and transcribes it into text.

    Args:
        language: A string indicating the language of the speech to be transcribed.

    Returns:
        A string containing the transcribed text.

    Raises:
        sr.RequestError: If there is an error with the API request.
        sr.UnknownValueError: If the speech could not be transcribed.
    """
    print("\n\nListening...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source) # causes a delay to sample for ambient noise

        audio = recognizer.listen(source)

        print("Recognizing...")
        transcript = recognizer.recognize_whisper(
            audio, language=language, model="base.en"
        )  # local whisper
        # transcript = client.audio.transcriptions.create(file=audio.get_wav_data(), model="whisper-1") # OAI whisper

        print(transcript)
        return transcript


def send_request(language: str, words: str) -> None:
    """
    Sends a request to the OpenAI API and speaks out the response.

    Args:
        language: A string indicating the language of the chat message.
        words: A string containing the chat message to be sent.

    Returns:
        None.

    Raises:
        openai.error.OpenAIError: If there is an error with the OpenAI API request.
    """
    if (
        words == "" or words == None or not any(c.isalpha() for c in words)
    ):  # if no words were said
        print("No words were said.")
        return

    conversation.append({"role": "user", "content": words})
    gen_response()


def get_gpt_stream():
    """
    Returns a stream of GPT-3 responses. Is a generator function.
    """
    global conversation # for vibes

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation,
        stream=True,
    )
    responseSoFar = ""

    streamChunk = ""
    punctuation = ".?!,"
    for response in completion:
        # Process each chunk of response
        newChunk = response.choices[0].delta.content
        if newChunk:
            print(f"newChunk: {newChunk}")

            responseSoFar += newChunk
            streamChunk += newChunk

            if any(c in punctuation for c in newChunk):
                print("yielding streamChunk: ", streamChunk)
                yield streamChunk
                streamChunk = ""

    conversation.append({"role": "assistant", "content": responseSoFar})
    print(conversation)


def gen_response() -> None:
    """
    Generates response and audio from a string of text.

    Args:
        words: A string containing the text to be converted to audio.

    Returns:
        None.

    Raises:
        openai.error.OpenAIError: If there is an error with the OpenAI API request.
    """
    print("Streaming audio...")
    audio_stream = elevenlabs.generate(
        text=get_gpt_stream(), 
        voice="Nicole", 
        stream=True 
    )
    
    elevenlabs.stream(audio_stream)

    # for response in get_gpt_stream():
    #     print("response: ", response)
    #     audio_stream = elevenlabs.generate(
    #         text=response, 
    #         voice="Nicole", 
    #         stream=True 
    #     )
    #     elevenlabs.stream(audio_stream)


if __name__ == "__main__":
    language = "english"

    while True:
        input_words = transcribe_speech(language)
        send_request(language, input_words)
