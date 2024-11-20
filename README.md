# cltl-dialogueclassification

## Description

Detects dialogue acts in texts and annotates the signals with the dialogue act labels and scores.
The annotations are pushed to the event bus and can be taken up for further processing.

We implemented three dialogue act classifiers:

1) Deberta fine-tuned with the SILICONE data set:

Based on: https://huggingface.co/diwank/silicone-deberta-pair

2) RoBERTa fine-tined with the MIDAS data set:

Based on: https://github.com/DianDYu/MIDAS_dialog_act

3) XLM-RoBERTa fine-tuned with the MIDAS data set:

Based on: https://github.com/DianDYu/MIDAS_dialog_act

## Getting started

### Prerequisites

This repository uses Python >= 3.9

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install -e .
    ```
2. Download the fine-tuned RoBERTA model from:

https://vu.data.surfsara.nl/index.php/s/xLou1DPl739Lbq6

and put the file "classifier.pt" in the directory:

resources/midas-da-roberta

Alternatively, download the XLM-roberta from: 

https://vu.data.surfsara.nl/index.php/s/dw0YCJAVFM870DT

and put the "pytorchmodel.bin" in the directory:

resources/midas-da-xlmroberta

### Usage

To apply this to emissor conversations:

    ```bash
        python3 examples/annotato_emissor_conversation_with_emotions.py --emissor "../data/emissor"
    ```

For using this repository as a package different project and on a different virtual environment, you may

- install a published version from PyPI:

    ```bash
    pip install cltl.dialogue_act_classification
    ```

- or, for the latest snapshot, run:

    ```bash
    pip install git+git://github.com/leolani/cltl-dialogueclassification.git@main
    ```

Then you can import it in a python script as:

    import cltl.dialogue_act_classification

To test the classifier run:

```commandline
PYTHONPATH=src python -m unittest
```

## References:
- Chapuis, Emile, Pierre Colombo, Matteo Manica, Matthieu Labeau, and Chloe Clavel. "Hierarchical pre-training for sequence labelling in spoken dialog." arXiv preprint arXiv:2009.11152 (2020).
- Yu, Dian, and Zhou Yu. "Midas: A dialog act annotation scheme for open domain human machine spoken conversations." arXiv preprint arXiv:1908.10023 (2019).
- 

## Integration in the Leolani event-bus

Can be integrated in the event-bus and to generate annotations in EMISSOR through a service.py that is included.
In the configuration file of the event-bus,the input and output topics need to specified as well as the emotion detectors.



