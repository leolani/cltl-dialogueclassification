import abc
import dataclasses
from typing import Optional, Any, List, Union
from enum import Enum, auto

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



@dataclasses.dataclass
class DialogueAct:
    """
    Information about a Dialogue Act.
    """
    type: str
    value: str
    confidence: Optional[float]



class DialogueActClassifier(abc.ABC):
    """Abstract DialogueActClassifier Object
    Call any of the modality specific emotion extraction function.
    """

    def _extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
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

