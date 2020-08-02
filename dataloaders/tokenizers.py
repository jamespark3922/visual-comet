from transformers import GPT2Tokenizer

class VisualCometTokenizer(GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 begin_img="<|b_img|>",
                 end_img="<|e_img|>",
                 begin_event="<|b_ev|>",
                 end_event="<|e_ev|>",
                 begin_place="<|b_pl|>",
                 end_place="<|e_pl|>",
                 begin_inferences={'before': "<|before|>", 'intent': "<|intent|>", 'after': "<|after|>"},
                 end_inference="<|e_in|>",
                 **kwargs):
        super(VisualCometTokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        self.begin_img = begin_img
        self.end_img = end_img
        self.begin_event = begin_event
        self.end_event = end_event
        self.begin_place = begin_place
        self.end_place = end_place
        self.begin_inferences = begin_inferences
        self.end_inference = end_inference
        self.det_tokens = ['<|det%d|>' % i for i in range(50)]
        self.add_special_tokens({
            "additional_special_tokens": [self.begin_img, self.end_img, self.begin_event, self.end_event,
                                          self.begin_place, self.end_place, self.end_inference]
                                         + list(self.begin_inferences.values()) + self.det_tokens
        })

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = super().decode(token_ids, False, clean_up_tokenization_spaces)
        tokens2remove = [self.begin_img, self.end_img, self.begin_event, self.end_event,
                         self.end_place, self.end_inference, self.unk_token] + list(self.begin_inferences.values())
        if skip_special_tokens:
            for t in tokens2remove:
                text = text.replace(t, ' ')
        idx = text.find(self.end_inference)
        if idx != -1:
            text = text[:idx]
        return text.strip()