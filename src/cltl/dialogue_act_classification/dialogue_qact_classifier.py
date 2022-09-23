import logging
import time
from typing import Any, List

from transformers import pipeline


from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

logger = logging.getLogger(__name__)

_MODEL_NAME = ""
_THRESHOLD = 0.5

#

class DialogueActDetector(DialogueActClassifier):
    def __init__(self, model: str = _MODEL_NAME):
        self.dialogue_act_pipeline = pipeline('sentiment-analysis',  model=model, return_all_scores=True)


    def extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
        if not utterance:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()

        acts = []

        response = self.dialogue_act_pipeline(utterance)

        self._log_results(acts, response, start)

        return acts


    def _log_results(self, acts, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("Dialogue acts detected: %s", [act.value for act in acts])



if __name__ == "__main__":
    utterance = "I love cats."
    analyzer = DialogueActDetector()
    acts = analyzer.extract_dialogue_act(utterance)
    act_json ={}
    for act in acts:
        act_json.update({'type': act.type, 'value':act.value, 'confident': act.confidence})
        print(act_json)
