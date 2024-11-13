import logging
from typing import List

from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import class_source

from cltl.dialogue_act_classification.api import DialogueActClassifier
from cltl_service.dialogue_act_classification.schema import DialogueActClassificationEvent

logger = logging.getLogger(__name__)


class DialogueActClassificationService:
    @classmethod
    def from_config(cls, extractor: DialogueActClassifier, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.dialogue_act_classification.events")

#@TODO chnge topic_inputs into topic_inpu and upadte a;; config files accordingly
        return cls(config.get("topic_inputs", multi=True), config.get("topic_output"),
                   config.get("topic_intention"), config.get("intentions", multi=True),
                   extractor, event_bus, resource_manager)

    def __init__(self, input_topics: List[str], output_topic: str,
                 intention_topic: str, intentions: List[str], extractor: DialogueActClassifier,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._extractor = extractor

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topics = input_topics
        self._output_topic = output_topic

        self._intention_topic = intention_topic if intention_topic else None
        self._intentions = set(intentions) if intentions else {}
        self._active_intentions = {}

        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker(self._input_topics, self._event_bus, provides=[self._output_topic],
                                         intentions=self._intentions, intention_topic=self._intention_topic,
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=4, name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[TextSignalEvent]):
        utterance= event.payload.signal.text
        acts = self._extractor.extract_dialogue_act(utterance)
        source = class_source(self._extractor)

        act_event = DialogueActClassificationEvent.create_dialogue_act_mentions(event.payload.signal, acts, source)
        self._event_bus.publish(self._output_topic, Event.for_payload(act_event))
