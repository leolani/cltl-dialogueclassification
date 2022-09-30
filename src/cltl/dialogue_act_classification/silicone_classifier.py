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

class DialogueAct(Enum):
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

class DialogueActDetector(DialogueActClassifier):
    def __init__(self):
       self._dialogue_act_pipeline = pipeline('text-classification', model=_MODEL_NAME)
       # self._model = ClassificationModel("deberta_v2", "diwank/silicone-deberta-pair", use_cuda=False)

      # self._tokenizer = DebertaTokenizer.from_pretrained("diwank/silicone-deberta-pair")
      # self._model = DebertaModel.from_pretrained("diwank/silicone-deberta-pair")

    def _extract_dialogue_act(self, sentences: str) -> List[DialogueAct]:
        if not sentences:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()
        responses = self._dialogue_act_pipeline(sentences)
        results = []
        for response in responses:
            print(response)
            dialogueAct = DialogueAct(type="SILICONE", value=response["label"], confidence=response["score"])
            results.append(dialogueAct)
        self._log_results(response, start)
        return results


    def _log_results(self, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)


#     def _convert_to_label (self, predictions):
#         labels = -=
#         for pred in predictions:
#             = lambda n: [
#     ['acknowledge',
#      'answer',
#      'backchannel',
#      'reply_yes',
#      'exclaim',
#      'say',
#      'reply_no',
#      'hold',
#      'ask',
#      'intent',
#      'ask_yes_no'
#     ][i] for i in n
# ]


if __name__ == "__main__":
    sentences = ["I love cats", "Do you love cats?", "Yes, I do", "No, dogs"]
    analyzer = DialogueActDetector()
    response = analyzer._extract_dialogue_act(sentences)
    for sentence, act in zip(sentences, response):
        print(sentence, act)
