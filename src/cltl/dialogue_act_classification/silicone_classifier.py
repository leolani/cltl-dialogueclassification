import logging
import time
from enum import Enum, auto
from typing import List

from transformers import pipeline

from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

logger = logging.getLogger(__name__)

#Local copy of the model
#_MODEL_NAME = "../models/silicone-deberta-pair"
_MODEL_NAME = "diwank/silicone-deberta-pair"
_THRESHOLD = 0.5

class DialogueActType(Enum):
    acknowledge = auto()
    answer = auto()
    backchannel = auto()
    reply_yes = auto()
    exclaim = auto()
    say = auto()
    reply_no = auto()
    hold = auto()
    ask = auto()
    intent = auto()
    ask_yes_no = auto()
    none = auto()
#['acknowledge','answer', 'backchannel', 'reply_yes', 'exclaim','say', 'reply_no', 'hold', 'ask', 'intent','ask_yes_no']

class DialogueActDetector(DialogueActClassifier):
    def __init__(self):
       self._dialogue_act_pipeline = pipeline('text-classification', model=_MODEL_NAME)

    def _extract_dialogue_act(self, sentences: str) -> List[DialogueAct]:
        if not sentences:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()
        responses = self._dialogue_act_pipeline(sentences)
        results = []
        for response in responses:
            print(response)
            response  = self._convert_to_label(response)
            try:
                dialogueAct = DialogueAct(type="SILICONE", value=response["label"], confidence=response["score"])
                results.append(dialogueAct)
            except:
                print(response)
        self._log_results(response, start)
        return results


    def _log_results(self, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)


    def _convert_to_label (self, prediction):
         label_index = int(prediction['label'][-1])
         prediction['label'] = DialogueActType(label_index)._name_
         return prediction


if __name__ == "__main__":
    sentences = ["I love cats", "Do you love cats?", "Yes, I do", "No, dogs"]
    analyzer = DialogueActDetector()
    response = analyzer._extract_dialogue_act(sentences)

    for sentence, act in zip(sentences, response):
        print(sentence, act)
