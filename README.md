# cltl-dialogueclassification

## Description

This package contains the functionality to detect dialogue acts in texts and annotates the signals with the dialogue act labels and scores.
The annotations are pushed to the event bus and can be taken up for further processing:



## Getting started

### Prerequisites

This repository uses Python >= 3.7

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install -e .
    ```

### Usage

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
    

## References:
- https://paperswithcode.com/task/dialogue-act-classification/codeless

- https://github.com/topics/dialogue-act-recognition

- https://github.com/ColingPaper2018/DialogueAct-Tagger

- https://huggingface.co/wwbproj/empathic_conversations_dialog_acts

- https://huggingface.co/diwank/silicone-deberta-pair

## Integration in the Leolani event-bus

Can be integrated in the event-bus and to generate annotations in EMISSOR through a service.py that is included.
In the configuration file of the event-bus,the input and output topics need to specified as well as the emotion detectors.



