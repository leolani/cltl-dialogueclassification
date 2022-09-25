# cltl-dialogueclassification
Detects dialogue acts in texts and annotates the signals with the dialogue act labels and scores.
The annotations are pushed to the event bus and can be taken up for further processing:


https://paperswithcode.com/task/dialogue-act-classification/codeless

https://github.com/topics/dialogue-act-recognition

https://github.com/ColingPaper2018/DialogueAct-Tagger

https://huggingface.co/wwbproj/empathic_conversations_dialog_acts

https://huggingface.co/diwank/silicone-deberta-pair


References:
-

## Integration in the Leolani event-bus

Can be integrated in the event-bus and to generate annotations in EMISSOR through a service.py that is included.
In the configuration file of the event-bus,the input and output topics need to specified as well as the emotion detectors.



