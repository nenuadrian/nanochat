"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

import re
from tasks.common import Task, load_hub_dataset, render_mc

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_hub_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    @staticmethod
    def extract_choice(response, letters):
        """Pull a choice letter out of free-form generated text, or None.

        evaluate() below can assume a bare letter because the eval harness scores
        categorically -- it picks among the letters by likelihood. RL generates
        instead, so the response is whatever the model felt like emitting and may
        not be a letter at all. Returning None (-> reward 0) is the correct
        outcome for an unparseable answer, not an error.
        """
        s = response.strip()
        if s in letters:
            return s
        # Tolerate "A.", "(A)", "Answer: A". \b stops "A" matching inside a word
        # like "Apple". First match wins: the prompt asks for the letter only, so
        # the model is trained to lead with it.
        m = re.search(r"\b(" + "|".join(re.escape(l) for l in letters) + r")\b", s)
        return m.group(1) if m else None

    def reward(self, conversation, assistant_response):
        """Used during RL. Unlike evaluate(), never raises on a malformed answer."""
        letters = conversation["letters"]
        pred = self.extract_choice(assistant_response, letters)
        if pred is None:
            return 0.0
        return float(pred == conversation["messages"][-1]["content"])

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
