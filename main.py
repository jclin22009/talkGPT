import os
from openai import OpenAI
import speech_recognition as sr
import elevenlabs

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
conversation = [
    {
        "role": "system",
        "content": """You are Jason Lin, a college student answering his door. Respond to folks knocking on your door. Be concise, witty, and natural. Here is some info about you:
               
               You live in the dorm Toyon. You study computer science at Stanford. You love to build things. You made a motorized driving couch, and there is a robot arm in your room. You love to take photos. You like to drink tea.
               For most people coming to the room, just mention that you're currently busy but would love to chat later.
               """,
    }
]


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

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
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

    # gpt_stream = get_gpt_stream()
    # print("gpt_stream: ", gpt_stream)
    gen_audio()

    # conversation.append({"role": "assistant", "content": answer})
    print(f"conversation so far: {conversation}")


def get_gpt_stream():
    """
    Returns a stream of GPT-3 responses. Is a generator.
    """
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation,
        stream=True,
    )

    for response in completion:
        # Process each chunk of response
        print("resp:", response.choices[0].delta.content)
        yield response.choices[0].delta.content


def gen_audio() -> None:
    """
    Generates audio from a string of text.

    Args:
        words: A string containing the text to be converted to audio.

    Returns:
        None.

    Raises:
        openai.error.OpenAIError: If there is an error with the OpenAI API request.
    """
    print("Generating audio...")
    audio = elevenlabs.generate(
        text=get_gpt_stream(), voice="Nicole", stream=True  # for now
    )
    print("audio generated")
    elevenlabs.stream(audio)
    # elevenlabs.play(audio)


if __name__ == "__main__":
    language = "english"

    while True:
        input_words = transcribe_speech(language)
        send_request(language, input_words)
