from typing import List

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
#from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from cltl.dialogue_act_classification.api import DialogueActClassifier, DialogueAct

try:
    from tqdm import tqdm
except:
    # Only needed for training
    pass


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
    def __init__(self, model_path, XLM=True):
        self._device = torch.device('cpu')

        if XLM:
            # Works for XLM Roberta for 100 languages
            #self._tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
           # self._model = RobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=len(_LABELS))
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=len(_LABELS))
            #self.load_state_dict(torch.load(model_path, map_location=self._device))

        else:
         #  Works for English Roberta Classifier
           # self._tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
           # self._model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(_LABELS))
            self._tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self._model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=len(_LABELS))

        #self._model.load_state_dict(torch.load(model_path, map_location=self._device), strict=False)

        self._model.to(self._device)

        self._label2id = _LABEL2ID
        self._id2label = _LABELS
        self._dialog =[""] ### initialise with an empty string to get started

    def _tokenize(self, strings):
        return self._tokenizer(strings, padding=True, return_tensors='pt').to(self._device)

    def _encode_labels(self, labels):
        for label in labels:
            if label not in self._label2id:
                self._label2id[label] = len(self._label2id)
                self._id2label[len(self._id2label)] = label

    # Only needed for training
    def fit(self, data, epochs=4, batch_size=32, lrate=1e-5):
        # Preprocess turns and index labels
        strings = [t0 + self._tokenizer.sep_token + t1 for t0, t1, _ in data]
        labels = [l for _, _, l in data]

        X = [self._tokenize(strings[i:i + batch_size]) for i in range(0, len(strings), batch_size)]
        y = [self._encode_labels(labels[i:i + batch_size]) for i in range(0, len(labels), batch_size)]

        # Setup optimizer and objective function
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lrate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            losses = []

            for X_batch, y_batch in tqdm(zip(X, y)):
                y_pred = self._model(**X_batch)
                loss = criterion(y_pred.logits, y_batch)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(np.mean(losses))

    def extract_dialogue_act(self, utterance: str)-> List[DialogueAct]:
        if not utterance:
            return []

        turn0 = self._dialog[-1]
        self._dialog.append(utterance)
        string = turn0 + self._tokenizer.sep_token + utterance
        X = self._tokenize([string])
        y = self._model(**X).logits.cpu().detach().numpy()
        label = self._id2label[np.argmax(y[0])]
        score = y[0][np.argmax(y[0])]
        dialogueAct = DialogueAct(type="MIDAS", value=label, confidence=float(score))
        ### Trying to normalize the scores, any ideas?
        #max = np.max(y[0])
        #min = np.min(y[0])
        #scaled_scores = np.array([(x-min)/(max-min) for x in y[0]])
        return [dialogueAct]


if __name__ == "__main__":
    sentences_en = ["I love cats", "Do you love cats?","Yes, I do", "Do you love cats?", "No, dogs"]
    sentences_nl = ["Ik ben dol op katten", "Hou jij van katten?","Ja, ik ben dol op ze", "Hou jij van katten?", "Nee, honden"]
   # model_path = "/Users/piek/Desktop/d-Leolani/leolani-models/dialogue_models/midas-da-xlmroberta"
    model_path = "/Users/piek/Desktop/d-Leolani/tutorials/test22/cltl-text-to-ekg-app/app/py-app/resources/midas-da-xlmroberta"
#   model_path="../../../resources/midas-da-roberta/classifier.pt"
#   model_path="../../../resources/midas-da-bert/midas-da-bert.bin"
    analyzer = MidasDialogTagger(model_path=model_path)

    for sentence in sentences_en+sentences_nl:
        response = analyzer.extract_dialogue_act(sentence)
        print(sentence, response)
