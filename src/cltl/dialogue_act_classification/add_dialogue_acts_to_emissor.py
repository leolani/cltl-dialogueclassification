import logging

from emissor.persistence import ScenarioStorage
from cltl.dialogue_act_classification.midas_classifier import MidasDialogTagger
from cltl_service.dialogue_act_classification.schema import DialogueActClassificationEvent
from emissor.persistence.persistence import ScenarioController
from emissor.processing.api import SignalProcessor
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, Mention, Annotation, Signal
from typing import Tuple, Mapping, Iterable
import uuid
from tqdm import tqdm
logger = logging.getLogger(__name__)



class DialogueActAnnotator (SignalProcessor):

    def __init__(self, model_path, XLM=True):
        """ an evaluator that will use reference metrics to approximate the quality of a conversation, across turns.
        params
        returns: None
        """
        self._classifier= MidasDialogTagger(model_path=model_path, XLM=XLM)
        self._max_text_length=514


    def process_signal(self, scenario: ScenarioController, signal: Signal):
        if not signal.modality == Modality.TEXT:
            return
        annotation = self.annotate(signal)
        print(annotation)

    def annotate(self, textSignal):
        utterance = textSignal.text
        if len(utterance)> self._max_text_length:
            utterance=utterance[:self._max_text_length]
        acts = self._classifier.extract_dialogue_act(utterance)
        print(textSignal.text, acts)
        act_event = DialogueActClassificationEvent.create_dialogue_act_mentions(textSignal, acts, "MIDAS")
        return act_event


if __name__ == "__main__":
    #    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/dialogue_models/midas-da-roberta/classifier.pt")
    #    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/dialogue_models/midas-da-bert/midas-da-bert.bin")
    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/dialogue_models/midas-da-xlmroberta/pytorch_model.bin"
    annotator = DialogueActAnnotator(model_path=model_path, XLM=True)
    scenario_folder = "/Users/piek/Desktop/d-Leolani/tutorials/test10/leolani-text-to-ekg/app/py-app/storage/emissor"

    scenario_storage = ScenarioStorage(scenario_folder)
    scenarios = list(scenario_storage.list_scenarios())
    print("Processing scenarios: ", scenarios)

    for scenario in scenarios:
        print('Processing scenario', scenario)
        scenario_ctrl = scenario_storage.load_scenario(scenario)
        signals = scenario_ctrl.get_signals(Modality.TEXT)
        for signal in signals:
            annotator.process_signal(scenario=scenario_ctrl, signal=signal)
