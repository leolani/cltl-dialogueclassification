import unittest

from cltl.dialogue_act_classification.dialogue_act_classifier import DialogueActDetector


class TestEmotions(unittest.TestCase):
    def setUp(self) -> None:
        self._dialogue_act_classifier = DialogueActDetector("diwank/silicone-deberta-pair")

    def test_analyze_text_with_emotion(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("I am so happy for you.")

        self.assertEqual(3, len(acts))
        self.assertEqual("assertion", acts[1].value)

    def test_analyze_empty(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("")
        self.assertEqual(0, len(acts))
