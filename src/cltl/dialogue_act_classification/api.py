import abc
import dataclasses
from enum import Enum, auto
from typing import Optional, List


# (0, 'acknowledge')
# (1, 'answer')
# (2, 'backchannel')
# (3, 'reply_yes')
# (4, 'exclaim')
# (5, 'say')
# (6, 'reply_no')
# (7, 'hold')
# (8, 'ask')
# (9, 'intent')
# (10, 'ask_yes_no')

class DialogueActType(Enum):
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
