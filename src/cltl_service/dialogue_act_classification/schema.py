import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, TextSignal, Annotation
from typing import Iterable

from cltl.dialogue_act_classification.api import DialogueAct


@dataclass
class DialogueActClassificationEvent(AnnotationEvent[Annotation[DialogueAct]]):
    @classmethod
    def create_text_mentions(cls, text_signal: TextSignal, acts: Iterable[DialogueAct]):
        return cls(cls.__name__, DialogueActClassificationEvent.to_mention(text_signal, acts))

    @staticmethod
    def to_mention(text_signal: TextSignal, acts: Iterable[DialogueAct]):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = text_signal.ruler
        annotations = [Annotation(DialogueAct.__class__.__name__, dialogueAct.value, __name__, timestamp_now())
                       for dialogueAct in acts]

        return Mention(str(uuid.uuid4()), [segment], annotations)

