
import queue
import threading
import numpy as np
import sounddevice as sd
import torch

from transformers import (
    AutoProcessor,
    AutoModelForRNNT,
    TextIteratorStreamer,
)

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
LANGUAGE = "en-US"
LOOKAHEAD = 6  # 560ms streaming latency

# Audio queue populated by microphone callback
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"\nAudio warning: {status}")

    audio_queue.put(indata[:, 0].copy())


def get_audio(num_samples, audio):
    """Collect microphone samples until enough audio is available."""
    while len(audio) < num_samples:
        audio = np.concatenate([audio, audio_queue.get()])

    return audio


def main():
    print("Loading Nemotron ASR...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForRNNT.from_pretrained(
        MODEL_ID,
        device_map="auto",
    )
    model.eval()

    processor.set_num_lookahead_tokens(LOOKAHEAD)

    sample_rate = processor.feature_extractor.sampling_rate

    print(f"Model loaded on: {model.device}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Streaming latency: {processor.streaming_latency_ms} ms")

    print("\nStarting microphone...")
    print("Speak into your microphone. Press CTRL+C to stop.\n")

    audio = np.empty(0, dtype=np.float32)

    # Start microphone capture
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        # Collect the initial audio chunk
        audio = get_audio(
            processor.num_samples_first_audio_chunk,
            audio,
        )

        first_chunk_inputs = processor(
            audio[:processor.num_samples_first_audio_chunk],
            sampling_rate=sample_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=LANGUAGE,
            return_tensors="pt",
        )

        first_chunk_inputs = first_chunk_inputs.to(
            model.device,
            dtype=model.dtype,
        )

        def input_features_generator():
            # First chunk
            yield first_chunk_inputs.input_features[
                :, :processor.num_mel_frames_first_audio_chunk, :
            ]

            mel_frame_idx = processor.num_mel_frames_first_audio_chunk

            hop_length = processor.feature_extractor.hop_length
            n_fft = processor.feature_extractor.n_fft

            start_idx = mel_frame_idx * hop_length - n_fft // 2

            # Continuously process new microphone audio
            while True:
                end_idx = start_idx + processor.num_samples_per_audio_chunk

                audio = get_audio(end_idx, audio)

                inputs = processor(
                    audio[start_idx:end_idx],
                    sampling_rate=sample_rate,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    language=LANGUAGE,
                    return_tensors="pt",
                )

                inputs = inputs.to(model.device, dtype=model.dtype)

                yield inputs.input_features

                mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                start_idx = mel_frame_idx * hop_length - n_fft // 2

        # Stream generated text
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_special_tokens=True,
        )

        generate_kwargs = {
            **first_chunk_inputs,
            "input_features": input_features_generator(),
            "streamer": streamer,
        }

        # Run inference separately so microphone capture continues
        thread = threading.Thread(
            target=model.generate,
            kwargs=generate_kwargs,
            daemon=True,
        )

        thread.start()

        print("LIVE TRANSCRIPTION:\n")

        for text_chunk in streamer:
            print(text_chunk, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping transcription...")
