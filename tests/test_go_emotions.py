import unittest

from cltl.emotion_extraction.emotion_mappings import EmotionType
from cltl.dialogue_act_classification.dialogue_qact_classifier import GoEmotionDetector


class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._emotion_extractor = GoEmotionDetector()

    def test_analyze_text_with_emotion(self):
        emotions = self._emotion_extractor.extract_text_emotions("I am so hapy for you.")

        self.assertEqual(3, len(emotions))

        self.assertEqual(EmotionType.GO, emotions[0].type)
        self.assertEqual("amusement", emotions[0].value)
        self.assertEqual(EmotionType.EKMAN, emotions[1].type)
        self.assertEqual("joy", emotions[1].value)
        self.assertEqual(EmotionType.SENTIMENT, emotions[2].type)
        self.assertEqual("positive", emotions[2].value)

    def test_analyze_empty(self):
        emotions = self._emotion_extractor.extract_text_emotions("")
        self.assertEqual(0, len(emotions))
