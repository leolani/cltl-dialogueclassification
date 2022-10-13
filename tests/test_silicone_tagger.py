import unittest

from cltl.dialogue_act_classification.silicone_classifier import DialogueActDetector


class DialogueActDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dialogue_act_classifier = DialogueActDetector()

    def test_analyze_opinion(self):
        acts = self._dialogue_act_classifier._extract_dialogue_act("I am so happy for you.")

        self.assertEqual(1, len(acts))
        self.assertEqual("SILICONE", acts[0].type)
        self.assertEqual("acknowledge", acts[0].value)

    def test_analyze_empty(self):
        acts = self._dialogue_act_classifier._extract_dialogue_act("")

        # TODO
        self.assertEqual(0, len(acts))
