from typing import List
import os
from transformers import pipeline, RobertaForSequenceClassification, AutoTokenizer
from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

# based on:
#https://github.com/DianDYu/MIDAS_dialog_act

_LABELS={0: 'open_question_factual',
          1: 'pos_answer',
          2: 'command',
          3: 'opinion',
          4: 'statement',
          5: 'back-channeling',
          6: 'yes_no_question',
          7: 'appreciation',
          8: 'other_answers',
          9: 'thanking',
          10: 'open_question_opinion',
          11: 'hold',
          12: 'closing',
          13: 'comment',
          14: 'neg_answer',
          15: 'complaint',
          16: 'abandon',
          17: 'dev_command',
          18: 'apology',
          19: 'nonsense',
          20: 'other',
          21: 'opening',
          22: 'respond_to_apology'}


_LABEL2ID ={'open_question_factual': 0,
            'pos_answer': 1,
            'command': 2,
            'opinion': 3,
            'statement': 4,
            'back-channeling': 5,
            'yes_no_question': 6,
            'appreciation': 7,
            'other_answers': 8,
            'thanking': 9,
            'open_question_opinion': 10,
            'hold': 11, 'closing': 12,
            'comment': 13,
            'neg_answer': 14,
            'complaint': 15,
            'abandon': 16,
            'dev_command': 17,
            'apology': 18,
            'nonsense': 19,
            'other': 20,
            'opening': 21,
            'respond_to_apology': 22}


class MidasDialogTagger(DialogueActClassifier):
    def __init__(self, model_path):
        abs_model_path = os.path.abspath(os.path.expanduser(model_path))
        self._tokenizer = AutoTokenizer.from_pretrained(abs_model_path, local_files_only=True)
        self._model = RobertaForSequenceClassification.from_pretrained(abs_model_path, local_files_only=True, use_safetensors=True, num_labels=len(_LABELS))
        self._pipeline = pipeline("text-classification", model=self._model, tokenizer=self._tokenizer)
        self._label2id = _LABEL2ID
        self._id2label = _LABELS
        self._dialog =[""] ### initialise with an empty string to get started


    def extract_dialogue_act(self, utterance: str)-> List[DialogueAct]:
        if not utterance:
            return []
        turn0 = self._dialog[-1]
        self._dialog.append(utterance)
        string = turn0 + self._tokenizer.sep_token + utterance
        result = self._pipeline(string)
        dialogueAct = DialogueAct(type="MIDAS", value=result[0]['label'], confidence=float(result[0]['score']))
        return [dialogueAct]

if __name__ == "__main__":
    sentences_en = ["I love cats", "Do you love cats?","Yes, I do", "Do you love cats?", "No, dogs"]
    sentences_nl = ["Ik ben dol op katten", "Hou jij van katten?","Ja, ik ben dol op ze", "Hou jij van katten?", "Nee, honden"]
    model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/dialogue_models/midas-da-xlmroberta"
    analyzer = MidasDialogTagger(model_path=model_path)
    for sentence in sentences_en+sentences_nl:
        response = analyzer.extract_dialogue_act(sentence)
        print(sentence, response)
