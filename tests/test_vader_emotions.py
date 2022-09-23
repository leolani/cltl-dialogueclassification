import unittest

from cltl.emotion_extraction.emotion_mappings import EmotionType
from cltl.emotion_extraction.utterance_vader_sentiment_extractor import VaderSentimentDetector


class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._emotion_extractor = VaderSentimentDetector()

    def test_analyze_text_with_emotion(self):
        emotions = self._emotion_extractor.extract_text_emotions("I am so hapy for you.")

        self.assertEqual(1, len(emotions))

        self.assertEqual(EmotionType.SENTIMENT, emotions[0].type)
        self.assertEqual("neutral", emotions[0].value)

    def test_analyze_empty(self):
        emotions = self._emotion_extractor.extract_text_emotions("")
        self.assertEqual(0, len(emotions))
