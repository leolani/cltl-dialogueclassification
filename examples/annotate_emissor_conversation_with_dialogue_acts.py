import argparse
from emissor.persistence import ScenarioStorage
from emissor.representation.scenario import Modality, Signal
from cltl.dialogue_act_classification.add_dialogue_acts_to_emissor import DialogueActAnnotator
import os

def main(emissor: str, model_path="../resources/midas-da-xlmroberta/pytorch_model.bin"):
    annotator = DialogueActAnnotator(model_path=model_path)
    scenario_folder = emissor
    scenario_storage = ScenarioStorage(scenario_folder)
    scenarios = list(scenario_storage.list_scenarios())
    print("Processing scenarios: ", scenarios)
    for scenario in scenarios:
        print('Processing scenario', scenario)
        scenario_ctrl = scenario_storage.load_scenario(scenario)
        signals = scenario_ctrl.get_signals(Modality.TEXT)
        for signal in signals:
            annotator.process_signal(scenario=scenario_ctrl, signal=signal)
        #### Save the modified scenario to emissor
        scenario_storage.save_scenario(scenario_ctrl)

### How to run: python3 examples/annotato_emissor_conversation_with_emotions.py --emissor "../data/emissor"

if __name__ == '__main__':
    default = "../data/emissor"
    model="../resources/midas-da-xlmroberta/pytorch_model.bin"
    parser = argparse.ArgumentParser(description='Annotate emissor with emotions')
    parser.add_argument('--emissor', type=str, required=True, help="Path to the folder with emissor scenarios", default=default)
    parser.add_argument('--model_path', type=str, required=True, help="Path to the MIDAS DA XLMRoBERTa model", default=model)
    args, _ = parser.parse_known_args()
    folder = os.path.exists(args.emissor)
    if not os.path.exists(args.emissor):
        raise ValueError("The folder %s does not exists. The --emissor argument should point to a folder that contains the scenarios to annotate", args.emissor)

    main(args.emissor.strip())
