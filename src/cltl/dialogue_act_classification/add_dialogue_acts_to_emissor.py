import logging
import argparse
import sys
from emissor.persistence import ScenarioStorage
from cltl.dialogue_act_classification.midas_classifier import MidasDialogTagger
from cltl_service.dialogue_act_classification.schema import DialogueActClassificationEvent
from emissor.persistence.persistence import ScenarioController
from emissor.processing.api import SignalProcessor
from emissor.representation.scenario import Modality, Signal
logger = logging.getLogger(__name__)

class DialogueActAnnotator (SignalProcessor):

    def __init__(self, model, model_name, XLM=True):
        """ an evaluator that will use reference metrics to approximate the quality of a conversation, across turns.
        params
        returns: None
        """
        self._classifier= MidasDialogTagger(model_path=model, XLM=XLM)
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

def main(emissor_path:str, scenario:str,  model:str, model_name:str):
    annotator = DialogueActAnnotator(model=model, model_name=model_name, XLM=True)
    scenario_storage = ScenarioStorage(emissor_path)
    scenario_ctrl = scenario_storage.load_scenario(scenario)
    signals = scenario_ctrl.get_signals(Modality.TEXT)
    for signal in signals:
        annotator.process_signal(scenario=scenario_ctrl, signal=signal)
    #### Save the modified scenario to emissor
    scenario_storage.save_scenario(scenario_ctrl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical evaluation emissor scenario')
    parser.add_argument('--emissor-path', type=str, required=False, help="Path to the emissor folder", default='')
    parser.add_argument('--scenario', type=str, required=False, help="Identifier of the scenario", default='')
    parser.add_argument('--model', type=str, required=False, help="Path to the fine-tuned model", default='')
    parser.add_argument('--model_name', type=str, required=False, help="Name of the model for the annoation", default='MIDAS')

    args, _ = parser.parse_known_args()
    print('Input arguments', sys.argv)
    main(emissor_path=args.emissor_path,
         scenario=args.scenario,
         model=args.model,
         model_name=args.model_name)
