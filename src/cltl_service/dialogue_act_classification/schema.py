import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, TextSignal, Annotation, class_type
from typing import Iterable

from cltl.dialogue_act_classification.api import DialogueAct


@dataclass
class DialogueActClassificationEvent(AnnotationEvent[Annotation[DialogueAct]]):
    @classmethod
    def create_dialogue_act_mentions(cls, text_signal: TextSignal, acts: Iterable[DialogueAct], source: str):
        return cls(class_type(cls), [DialogueActClassificationEvent.to_mention(text_signal, acts, source)])

    @staticmethod
    def to_mention(text_signal: TextSignal, acts: Iterable[DialogueAct], source: str):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = text_signal.ruler
        annotations = [Annotation(class_type(DialogueAct), dialogueAct, source, timestamp_now())
                       for dialogueAct in acts]

        return Mention(str(uuid.uuid4()), [segment], annotations)

