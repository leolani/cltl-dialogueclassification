import logging
import time
from enum import Enum
from typing import List

from transformers import pipeline

from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

logger = logging.getLogger(__name__)

#Local copy of the model
#_MODEL_NAME = "../models/silicone-deberta-pair"
_MODEL_NAME = "diwank/silicone-deberta-pair"
_THRESHOLD = 0.5


class SiliconeDialogueAct(Enum):
    # KEEP ORDER!
    acknowledge = 0
    answer = 1
    backchannel = 2
    reply_yes = 3
    exclaim = 4
    say = 5
    reply_no = 6
    hold = 7
    ask = 8
    intent = 9
    ask_yes_no = 10
    none = 11


class SiliconeDialogueActClassifier(DialogueActClassifier):
    def __init__(self):
       self._dialogue_act_pipeline = pipeline('text-classification', model=_MODEL_NAME)

    def extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
        if not utterance:
            return []

        logger.debug(f"Classify dialogue act...")
        start = time.time()
        responses = self._dialogue_act_pipeline(utterance)
        logger.info("Found %s dialogue acts in %s sec", len(responses), time.time() - start)
        logger.debug("Dialogue act values: %s", responses)

        return [self._to_dialogue_act(response) for response in responses]

    def _to_dialogue_act(self, prediction):
        label_index = int(prediction['label'][-1])
        label = SiliconeDialogueAct(label_index)._name_

        return DialogueAct(type="SILICONE", value=label, confidence=float(prediction["score"]))


if __name__ == "__main__":
    sentences = ["I love cats", "Do you love cats?", "Yes, I do", "No, dogs"]
    analyzer = SiliconeDialogueActClassifier()
    response = analyzer.extract_dialogue_act(sentences)

    for sentence, act in zip(sentences, response):
        print(sentence, act)
