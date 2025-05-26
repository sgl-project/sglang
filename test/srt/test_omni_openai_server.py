"""
Usage:
python3 -m unittest test_omni_openai_server
"""

import unittest

from test_vision_openai_server_common import *


# Omni Models
class TestOpenAIOmniServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-Omni-7B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        cls.base_url += "/v1"

    def prepare_audio_messages(self, prompt, audio_file_name):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"{audio_file_name}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        return messages

    def get_audio_response(self, url: str, prompt, category):
        audio_file_path = self.get_or_download_file(url)
        client = openai.Client(api_key="sk-123456", base_url=self.base_url)

        messages = self.prepare_audio_messages(prompt, audio_file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio {category} response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        return audio_response

    def verify_speech_recognition_response(self, text):
        text = text.lower()
        assert "thank you" in text
        assert "it's a privilege to be here" in text
        assert "leader" in text
        assert "science" in text
        assert "art" in text

    def _test_audio_speech_completion(self):
        # a fragment of Trump's speech
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            # "I have an audio sample. Please repeat the person's words",
            "Repeat exactly what does the person say in the audio. Be exact",
            category="speech",
        )
        self.verify_speech_recognition_response(audio_response)

    def _test_audio_ambient_completion(self):
        # bird song
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "Please listen to the audio snippet carefully and transcribe the content.",
            "ambient",
        )
        assert "bird" in audio_response

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()

    def test_mixed_modality_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": AUDIO_TRUMP_SPEECH_URL},
                    },
                    {
                        "type": "text",
                        "text": "I have an image and audio, which are not related at all. Please:  1. Describe the image in a sentence, 2. Repeat the exact words from the audio I provided. Be exact",
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        text = response.choices[0].message.content

        print("-" * 30)
        print(f"Mixed modality response:\n{text}")
        print("-" * 30)

        self.verify_single_image_response(response=response)
        self.verify_speech_recognition_response(text=text)


class TestMinicpmoServer(TestOpenAIOmniServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.7",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
    unittest.main()
