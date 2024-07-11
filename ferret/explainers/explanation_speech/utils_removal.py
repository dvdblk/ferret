from pydub import AudioSegment
import os
import numpy as np


def remove_specified_words(audio, words, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    from copy import deepcopy

    audio_removed = deepcopy(audio)

    a, b = 100, 40

    from IPython.display import display

    for word in words:
        start = int(word["start"] * 1000)
        end = int(word["end"] * 1000)

        before_word_audio = audio_removed[: start - a]
        after_word_audio = audio_removed[end + b :]

        word_duration = (end - start) + a + b

        if removal_type == "nothing":
            replace_word_audio = AudioSegment.empty()
        elif removal_type == "silence":
            replace_word_audio = AudioSegment.silent(duration=word_duration)
        elif removal_type == "white noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]
        elif removal_type == "pink noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed


def remove_word(audio, word, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    WARNING: if `word["start"] * 1000 - a` is negative, the audio is actually
             traversed FROM SOME POINT UNTIL ITS END (like `l[-10:]`
             actually takes the last 10 entries of the list `l`). Therefore if
             the difference is negative, we effectively use `a=0`.

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    a, b = 100, 40

    # Convert from seconds (as returned by WhisperX) to milliseconds (as
    # required to index PyDub `AudioSegment` objects).
    word_start_ms = word["start"] * 1000
    word_end_ms = word["end"] * 1000

    # If we risk reading the audio segment from the end (difference is
    # negative) set the offset `a` to zero to avoid that.
    if word_start_ms - a < 0:
        a = 0

    before_word_audio = audio[:word_start_ms - a]
    after_word_audio = audio[word_end_ms + b :]
    word_duration = (word_end_ms - word_start_ms) + a + b

    if removal_type == "nothing":
        replace_word_audio = AudioSegment.empty()

    elif removal_type == "silence":
        replace_word_audio = AudioSegment.silent(duration=word_duration)

    elif removal_type == "white noise":
        sound_path = os.path.join(os.path.dirname(__file__), "white_noise.mp3")

        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

    elif removal_type == "pink noise":
        sound_path = os.path.join(os.path.dirname(__file__), "pink_noise.mp3")
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

    audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed
