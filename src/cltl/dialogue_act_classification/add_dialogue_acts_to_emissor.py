import logging
from emissor.persistence import ScenarioStorage
from cltl.dialogue_act_classification.midas_classifier import MidasDialogTagger
from cltl_service.dialogue_act_classification.schema import DialogueActClassificationEvent
from emissor.persistence.persistence import ScenarioController
from emissor.processing.api import SignalProcessor
from emissor.representation.scenario import Modality, Signal

logger = logging.getLogger(__name__)

class DialogueActAnnotator (SignalProcessor):

    def __init__(self, model_path, model_name):
        """ an evaluator that will use reference metrics to approximate the quality of a conversation, across turns.
        params
        returns: None
        """
        self._classifier= MidasDialogTagger(model_path=model_path)
        self._max_text_length=514
        self._model_name = model_name

    def process_signal(self, scenario: ScenarioController, signal: Signal):
        if not signal.modality == Modality.TEXT:
            return
        mention = self.annotate(signal)
        signal.mentions.append(mention)

    def annotate(self, textSignal):
        utterance = textSignal.text
        if len(utterance)> self._max_text_length:
            utterance=utterance[:self._max_text_length]
        acts = self._classifier.extract_dialogue_act(utterance)
        mention = DialogueActClassificationEvent.to_mention(textSignal, acts, self._model_name)
        return mention


    def remove_annotations(self, signal, annotation_source: [str]):
        keep_mentions = []
        for mention in signal.mentions:
            clear = False
            for annotation in mention.annotations:
                if annotation.source and annotation.source in annotation_source:
                    clear = True
                    break
            if not clear:
                keep_mentions.append(mention)
        signal.mentions = keep_mentions

    def process_all_scenarios(self, emissor_path:str, scenarios:[]):
        for scenario in scenarios:
            if not scenario.startswith("."):
                print(emissor_path, scenario)
                scenario_storage = ScenarioStorage(emissor_path)
                scenario_ctrl = scenario_storage.load_scenario(scenario)
                signals = scenario_ctrl.get_signals(Modality.TEXT)
                for signal in signals:
                    self.process_signal(scenario=scenario_ctrl, signal=signal)
                #### Save the modified scenario to emissor
                scenario_storage.save_scenario(scenario_ctrl)

