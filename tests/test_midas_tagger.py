import unittest
import importlib.resources as pkg_resources
import resources

from cltl.dialogue_act_classification.midas_classifier import MidasDialogTagger


class MidasDialogTaggerTest(unittest.TestCase):
    def setUp(self) -> None:
        resource_dir = pkg_resources.files(resources)
        model_path = resource_dir.joinpath('midas-da-roberta/classifier.pt')
        self._dialogue_act_classifier = MidasDialogTagger(model_path)

    def test_analyze_opinion(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("I am so happy for you.")

        self.assertEqual(1, len(acts))
        self.assertEqual("MIDAS", acts[0].type)
        self.assertEqual("opinion", acts[0].value)

    def test_analyze_empty(self):
        acts = self._dialogue_act_classifier.extract_dialogue_act("")

        self.assertEqual(0, len(acts))
