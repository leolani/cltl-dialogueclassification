import unittest

from parameterized import parameterized

from cltl.dialogue_act_classification.silicone_classifier import SiliconeDialogueActClassifier


class DialogueActDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dialogue_act_classifier = SiliconeDialogueActClassifier()

    @parameterized.expand([
        ("I love cats", ["say"]),
        ("Do you love cats?", ["ask"]),
        ("Yes, I do", ["reply_yes"]),
        ("No, I don't", ["answer"]),
    ])
    def test_analyze_utterances(self, utterance, expected):
        acts = self._dialogue_act_classifier.extract_dialogue_act(utterance)

        self.assertEqual(1, len(expected))
        self.assertTrue(all(act.type == "SILICONE" for act in acts))
        self.assertEqual(expected, [act.value for act in acts])

    def test_analyze_sequential(self):
        utterances = ["I love cats", "Do you love cats?", "Yes, I do", "Do you love dogs?", "No, I don't"]

        results = [self._dialogue_act_classifier.extract_dialogue_act(utterance) for utterance in utterances]

        # self.assertEqual(['say', 'ask_yes_no', 'reply_yes', "ask_yes_no", "reply_no"],
        self.assertEqual(['say', 'ask', 'reply_yes', "ask", "answer"],
                         [act.value for acts in results for act in acts])

    def test_analyze_empty(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("")

        self.assertEqual(0, len(acts))
