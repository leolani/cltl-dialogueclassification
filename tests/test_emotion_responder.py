import unittest

from cltl.emotion_extraction.api import Emotion
from cltl.emotion_extraction.emotion_mappings import EmotionType, EkmanEmotion, GoEmotion, Sentiment
from cltl.emotion_responder.emotion_responder import EmotionResponderImpl
from cltl.emotion_responder.emotion_sentences import _EKMAN_RESPONSES, _GO_RESPONSES, _SENTIMENT_RESPONSES


class TestResponder(unittest.TestCase):
    def setUp(self) -> None:
        self._responder = EmotionResponderImpl()

    def test_respond_to_go(self):
        for emotion in GoEmotion:
            response = self._responder.respond([Emotion(EmotionType.GO, emotion.name.lower(), 1.0)], "Piek")

            if emotion in _GO_RESPONSES:
                self.assertTrue("Piek" in response)
                self.assertTrue(any(phrase in response for phrase in _GO_RESPONSES[emotion]))

    def test_respond_to_ekman(self):
        for emotion in EkmanEmotion:
            response = self._responder.respond([Emotion(EmotionType.EKMAN, emotion.name.lower(), 1.0)], "Piek")

            if emotion in _EKMAN_RESPONSES:
                self.assertTrue("Piek" in response)
                self.assertTrue(any(phrase in response for phrase in _EKMAN_RESPONSES[emotion]))

    def test_respond_to_sentiment(self):
        for emotion in Sentiment:
            response = self._responder.respond([Emotion(EmotionType.SENTIMENT, emotion.name.lower(), 1.0)], "Piek")

            if emotion in _SENTIMENT_RESPONSES:
                self.assertTrue("Piek" in response)
                self.assertTrue(any(phrase in response for phrase in _SENTIMENT_RESPONSES[emotion]))

    def test_respond_to_empty(self):
        response = self._responder.respond("", "Piek")
        self.assertEqual("", response)
