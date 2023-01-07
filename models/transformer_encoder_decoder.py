import torch
import os
from base.base_model import BaseModel
from blocks.transformer import Encoder
from blocks.transformer import Decoder
from blocks.transformer.configs import TransformerModelConfig
from utils import Vocabulary, exists, keys_to_read, remove_decoration, fasta_sequence, sequence_to
from numpy import array
from typing import Optional, Union


class Transformer(BaseModel):
    """
    A standard encoder decoder transformer model.
    """
    def __init__(self, config):
        super(Transformer, self).__init__()

        # configs
        self.src_pad_idx = config.src_pad_idx
        self.trg_pad_idx = config.trg_pad_idx

        self.n_embd = config.n_embd
        # self.device = config.device

        # model
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.apply(self._init_weights)

    def forward(self, src, trg):
        input_mask = src != self.src_pad_idx
        enc_src = self.encoder(src, input_mask)

        target_mask = trg != self.src_pad_idx
        out = self.decoder(
            trg,
            enc_src,
            input_mask=input_mask,
            target_mask=target_mask,
            shared_embedding=self.encoder.embedding
        )

        return out

    @staticmethod
    def get_default_config():
        config_dict = {
            "src_pad_idx": 0,
            "src_vocab_size": 16,
            "trg_pad_idx": 0,
            "trg_vocab_size": 16,
            "block_size": 512,

            "n_embd": 512,

            "n_encoder_feedforward_layer": 1,
            "encoder_feedforward_forward_expansion": 4,
            "n_encoder_block": 4,

            "decoder_add_cross_attention_layer": True,
            "decoder_output_softmax_temperature": 1.0,
            "n_decoder_feedforward_layer": 1,
            "decoder_feedforward_forward_expansion": 4,
            "n_decoder_block": 4,

            "attention_type": "classic",
            "n_head": 16,
            "relative_position_embedding": True,
            "add_relative_position_to_values": True,
            "max_relative_position": 100,

            "encoder_feedforward_pdrop": 0.1,
            "decoder_feedforward_pdrop": 0.1,
            "attention_pdrop": 0.1,
            "attention_values_pdrop": 0.1,
            "embd_pdrop": 0.1
        }
        return TransformerModelConfig(config_dict)

    @torch.no_grad()
    def sample(self, src, **kwargs):
        """
        take a conditioning sequence of indices in src encode it and predict the next token in
        the target from the start token index, feeding the predictions back into the blocks each time.
        """
        # block_size = blocks.get_block_size()
        self.eval()
        vocab = kwargs["vocab"] if "vocab" in kwargs else None
        assert not exists(vocab), "provide a vocabulary instance to the sampler"
        temperature = kwargs["temperature"] if "temperature" in kwargs else 1.0
        sample = kwargs["sample"] if "sample" in kwargs else True
        top_k = kwargs["top_k"] if "top_k" in kwargs else False

        block_size = vocab.get_block_size()
        src_trunc = src if src.size(1) <= block_size else src[:, -block_size:]
        src_pad_idx = vocab.get_src_pad_idx()
        input_mask = src_trunc != src_pad_idx
        device = next(self.parameters()).device
        memory = self.encoder(src_trunc.type(torch.int), input_mask).to(device)

        strt_idx = vocab.get_trg_bos_idx()
        end_idx = vocab.get_trg_eos_idx()
        trgt_start = torch.tensor([[strt_idx]]).type(torch.int).to(device)
        # print(f"the start: {trgt_start}")
        while True:
            target_mask = trgt_start != src_pad_idx
            out = self.decoder(trgt_start, memory, input_mask, target_mask)

            out = self.generator(out[:, -1], temperature)

            if top_k is not None:
                v, _ = torch.topk(out, top_k)
                out[out < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert to probabilities
            probs = torch.exp(out)
            # print(probs)
            # sample from the distribution or take the most likely
            if sample:
                next_word = torch.multinomial(probs, num_samples=1)
            else:
                _, next_word = torch.topk(probs, k=1, dim=-1)
            # if next_word
            # append to the sequence and continue
            trgt_start = torch.cat(
                (trgt_start, torch.empty(1, 1).type_as(src_trunc.data).fill_(int(next_word))), dim=1
            )
            # if next_word == end_idx or trgt_start.shape[1] >= src.size(1):
            # tmp = trgt_start[0]
            # tmp = tmp[tmp != strt_idx]
            # tmp = tmp[tmp != src_pad_idx]
            # tmp = tmp[tmp != end_idx]
            #
            # if tmp.shape[0] >= src_trunc.size(1):
            #     break
            if next_word == end_idx:
                break
            # del tmp
        tmp_trgt_start = trgt_start.cpu()
        del trgt_start
        del memory
        torch.cuda.empty_cache()
        return tmp_trgt_start

    @torch.no_grad()
    def simulate_reads(self, reference_path, n_reads, output_file, vocab: Vocabulary):
        # retreive the token spliting mode
        kmers_mode = vocab.mode
        k = vocab.k

        reference = fasta_sequence(reference_path)
        reference = reference[:145] + 'GGG' + reference[145:164] + 'A' + reference[164:]
        # reference = dpu.generate_random_sequence(700)
        # print(reference)
        reference = sequence_to(reference, k, kmers_mode)
        # reference = dpu.fasta_sequence_to(config["reference"], k, kmers_mode)
        ref_tok = ('<BOS> ' + reference + ' <EOS>').split()
        # reference = dpu.sequence_to_words(data.ref[0][:145] + 'GGG', k)
        # ref_tok = reference.split()
        device = next(self.parameters()).device
        with open(output_file, "w") as f:
            x = torch.tensor([vocab.kmertoi[s] for s in ref_tok], dtype=torch.long)[None, ...].to(device)
            for r in range(n_reads):
                y = array(
                    self.sample(
                        x,
                        vocab=vocab,
                        temperature=1.0,
                        sample=True,
                        top_k=None
                    ).cpu()
                )

                read = keys_to_read(y[0], vocab, kmers_mode)
                read = remove_decoration(read)

                print(f"read {r}: {len(read)}")

                f.write(f">read{r}\n")
                f.write(f"{''.join(read)}\n")




