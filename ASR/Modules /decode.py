import sentencepiece as spm

class SentencePieceTransform:
    """Maps subwords to integers and vice versa using SentencePiece"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_int(self, text):
        """ Use the SentencePiece tokenizer to convert text to an integer sequence """
        subwords = self.sp.EncodeAsPieces(text.lower())
        return [self.sp.PieceToId(subword) for subword in subwords]

    def int_to_text(self, labels):
        """ Use the SentencePiece tokenizer to convert integer labels to a text sequence """
        return self.sp.decode(labels)

sentencepiece_transform_e = SentencePieceTransform("libri1000u.model")
sentencepiece_transform_c = SentencePieceTransform("aishell_unigram5000_model.model")


