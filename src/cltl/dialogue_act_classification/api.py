import abc
import dataclasses
from typing import Optional, Any, List, Union
from enum import Enum, auto


class DialogueActType(Enum):
    GO = auto()
    EKMAN = auto()
    FACE = auto()
    SENTIMENT = auto()

@dataclasses.dataclass
class DialogueAct:
    """
    Information about a Dialogue Act.
    """
    type: DialogueActType
    value: str
    confidence: Optional[float]



class DialogueActClassifier(abc.ABC):
    """Abstract DialogueActClassifier Object
    Call any of the modality specific emotion extraction function.
    """

    def extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
        """Recognize the dialogue act of a given utterance.

        Parameters
        ----------
        utterance : str
            The utterance to be analyzed.

        Returns
        -------
        List[DialogueAct]
            The DialogueAct extracted from the utterance.
        """
        raise NotImplementedError()

