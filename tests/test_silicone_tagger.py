import unittest

from parameterized import parameterized

from cltl.dialogue_act_classification.silicone_classifier import SiliconeDialogueActClassifier


class DialogueActDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dialogue_act_classifier = SiliconeDialogueActClassifier()

    @parameterized.expand([
        ("I love cats", ["say"]),
        # ("Do you love cats?", ["ask_yes_no"]),
        ("Do you love cats?", ["ask"]),
        ("Yes, I do", ["reply_yes"]),
        # ("No, I don't", ["reply_no"]),
        ("No, I don't", ["answer"]),
    ])
    def test_analyze_utterances(self, utterance, expected):
        acts = self._dialogue_act_classifier.extract_dialogue_act(utterance)

        self.assertEqual(1, len(expected))
        self.assertTrue(all(act.type == "SILICONE" for act in acts))
        self.assertEqual(expected, [act.value for act in acts])

    def test_analyze_sequential(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act(utterance)

        self.assertEqual(1, len(expected))
        self.assertTrue(all(act.type == "SILICONE" for act in acts))
        self.assertEqual(expected, [act.value for act in acts])

    def test_analyze_empty(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("")

        # TODO
        self.assertEqual(0, len(acts))
