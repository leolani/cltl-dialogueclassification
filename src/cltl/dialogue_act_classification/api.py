import abc
import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class DialogueAct:
    """
    Information about a Dialogue Act.
    """
    type: str
    value: str
    confidence: Optional[float]


class DialogueActClassifier(abc.ABC):
    """Classifier for the dialog act of an utterance."""

    def extract_dialogue_act(self, utterance: str) -> List[DialogueAct]:
        """Recognize the dialogue act of a given utterance.

        The result may depend on previous invocations of the method.

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

