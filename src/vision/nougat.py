import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel



class Nougat(object):
    def __init__(self, ):
        self.processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def __call__(self, image, cls=True):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # generate transcription (here we only generate 30 tokens)
        outputs = self.model.generate(
            pixel_values.to(self.device),
            min_length=1,
            max_new_tokens=2048,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        )

        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        sequence = self.processor.post_process_generation(sequence, fix_markdown=False)
        return sequence
