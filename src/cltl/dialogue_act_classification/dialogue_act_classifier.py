import logging
import time
from typing import Any, List

from transformers import TextClassificationPipeline, pipeline
# class transformers.DebertaConfig
# class transformers.DebertaTokenizer
# class transformers.DebertaModel
from transformers import DebertaTokenizer, DebertaModel
import torch

from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

logger = logging.getLogger(__name__)

# Local copy of the model
# _MODEL_NAME = "../models/silicone-deberta-pair"
_MODEL_NAME = "diwank/silicone-deberta-pair"
_THRESHOLD = 0.5


class DialogueActDetector(DialogueActClassifier):
    def __init__(self, model: str):
        self._dialogue_act_pipeline = pipeline('text-classification', model=model, top_k=3)
        # self._model = ClassificationModel("deberta_v2", "diwank/silicone-deberta-pair", use_cuda=False)
        # self._tokenizer = DebertaTokenizer.from_pretrained("diwank/silicone-deberta-pair")
        # self._model = DebertaModel.from_pretrained("diwank/silicone-deberta-pair")

    def extract_dialogue_act(self, sentences: str) -> List[DialogueAct]:
        if not sentences:
            return []

        logger.debug(f"sending utterance to server...")
        start = time.time()

        acts = []

        response = self._dialogue_act_pipeline(sentences)
        # response, raw = self._model.predict(sentences)
        # inputs = self._tokenizer(sentences, return_tensors="pt")
        # response =self._model(**inputs)
        self._log_results(acts, response, start)

        return acts

    def _log_results(self, acts, response, start):
        logger.info("got %s from server in %s sec", response, time.time() - start)
        logger.info("Dialogue acts detected: %s", [act.value for act in acts])


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
    analyzer = DialogueActDetector(_MODEL_NAME)
    response = analyzer.extract_dialogue_act(sentences[0])
    print(sentences, response)

# Expected output
# [{'label': 'LABEL_5', 'score': 0.7644516229629517},
#  {'label': 'LABEL_8', 'score': 0.9776815176010132},
#  {'label': 'LABEL_3', 'score': 0.5020651817321777},
#  {'label': 'LABEL_5', 'score': 0.5964227318763733}]
