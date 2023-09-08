
from typing import List, Optional
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import audio_preprocessing
import logging

preprocessor = audio_preprocessing.AudioToMelSpectrogramPreprocessor(features=80)
preprocessor.featurizer.dither = 0.0
preprocessor.featurizer.pad_to = 0






@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestep

        Returns:
            List with confidence scores. The length of the list is the same as `timestep`.
        """
        non_blank_frame_confidence = []
        # self.timestep can be a dict for RNNT
        timestep = self.timestep['timestep'] if isinstance(self.timestep, dict) else self.timestep
        if len(self.timestep) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestep:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestep]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()


@dataclass
class NBestHypotheses:
    """List of N best hypotheses"""

    n_best_hypotheses: Optional[List[Hypothesis]]


@dataclass
class HATJointOutput:
    """HATJoint outputs for beam search decoding

    hat_logprobs: standard HATJoint outputs as for RNNTJoint

    ilm_logprobs: internal language model probabilities (for ILM subtraction)
    """

    hat_logprobs: Optional[torch.Tensor] = None
    ilm_logprobs: Optional[torch.Tensor] = None


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """
    Obtained from https://github.com/espnet/espnet.

    Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def select_k_expansions(
    hyps: List[Hypothesis], topk_idxs: torch.Tensor, topk_logps: torch.Tensor, gamma: float, beta: int,
) -> List[Tuple[int, Hypothesis]]:
    """
    Obtained from https://github.com/espnet/espnet

    Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        topk_idxs: Indices of candidates hypothesis. Shape = [B, num_candidates]
        topk_logps: Log-probabilities for hypotheses expansions. Shape = [B, V + 1]
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(v)) for k, v in zip(topk_idxs[i], topk_logps[i])]
        k_best_exp_val = max(hyp_i, key=lambda x: x[1])

        k_best_exp_idx = k_best_exp_val[0]
        k_best_exp = k_best_exp_val[1]

        expansions = sorted(filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i), key=lambda x: x[1],)

        if len(expansions) > 0:
            k_expansions.append(expansions)
        else:
            k_expansions.append([(k_best_exp_idx, k_best_exp)])

    return k_expansions

def pack_hypotheses(hypotheses: List[Hypothesis], logitlen: torch.Tensor,) -> List[Hypothesis]:

    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)
        hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class ExportedModelGreedyBatchedRNNTInfer:
    def __init__(self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = None):
        self.encoder_model_path = encoder_model
        self.decoder_joint_model_path = decoder_joint_model
        self.max_symbols_per_step = max_symbols_per_step

        # Will be populated at runtime
        self._blank_index = None

    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output, encoded_lengths = self.run_encoder(audio_signal=audio_signal, length=length)

            if torch.is_tensor(encoder_output):
                encoder_output = encoder_output.transpose(1, 2)
            else:
                encoder_output = encoder_output.transpose([0, 2, 1])  # (B, T, D)
            logitlen = encoded_lengths

            inseq = encoder_output  # [B, T, D]
            hypotheses, timestamps = self._greedy_decode(inseq, logitlen)

            # Pack the hypotheses results
            packed_result = [Hypothesis(score=-1.0, y_sequence=[]) for _ in range(len(hypotheses))]
            for i in range(len(packed_result)):
                packed_result[i].y_sequence = torch.tensor(hypotheses[i], dtype=torch.long)
                packed_result[i].length = timestamps[i]

            del hypotheses

        return packed_result

    def _greedy_decode(self, x, out_len):
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        hidden = self._get_initial_states(batchsize)
        target_lengths = torch.ones(batchsize, dtype=torch.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long).numpy()
        if torch.is_tensor(x):
            last_label = torch.from_numpy(last_label).to(self.device)

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool).numpy()

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]

            if torch.is_tensor(f):
                f = f.transpose(1, 2)
            else:
                f = f.transpose([0, 2, 1])

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len
            # Start inner loop
            while not_blank and (self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = torch.tensor([self._blank_index] * batchsize, dtype=torch.int32).view(-1, 1)
                else:
                    if torch.is_tensor(last_label):
                        g = last_label.type(torch.int32)
                    else:
                        g = last_label.astype(np.int32)

                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = self.run_decoder_joint(f, g, target_lengths, *hidden)
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                if torch.is_tensor(logp):
                    v, k = logp.max(1)
                else:
                    k = np.argmax(logp, axis=1).astype(np.int32)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask |= k_is_blank

                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    if torch.is_tensor(blank_mask):
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    else:
                        blank_indices = blank_mask.astype(np.int32).nonzero()

                    if type(blank_indices) in (list, tuple):
                        blank_indices = blank_indices[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    if torch.is_tensor(k):
                        last_label = k.clone().reshape(-1, 1)
                    else:
                        last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    def _setup_blank_index(self):
        raise NotImplementedError()

    def run_encoder(self, audio_signal, length):
        raise NotImplementedError()

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        raise NotImplementedError()

    def _get_initial_states(self, batchsize):
        raise NotImplementedError()


class ONNXGreedyBatchedRNNTInfer(ExportedModelGreedyBatchedRNNTInfer):
    def __init__(self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = 10):
        super().__init__(
            encoder_model=encoder_model,
            decoder_joint_model=decoder_joint_model,
            max_symbols_per_step=max_symbols_per_step,
        )

        try:
            import onnx
            import onnxruntime
        except (ModuleNotFoundError, ImportError):
            raise ImportError(f"`onnx` or `onnxruntime` could not be imported, please install the libraries.\n")

        if torch.cuda.is_available():
            # Try to use onnxruntime-gpu
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
        else:
            # Fall back to CPU and onnxruntime-cpu
            providers = ['CPUExecutionProvider']

        onnx_session_opt = onnxruntime.SessionOptions()
        onnx_session_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_model = onnx.load(self.encoder_model_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.encoder_model = onnx_model
        self.encoder = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=providers, provider_options=onnx_session_opt
        )

        onnx_model = onnx.load(self.decoder_joint_model_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.decoder_joint_model = onnx_model
        self.decoder_joint = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=providers, provider_options=onnx_session_opt
        )

        logging.info("Successfully loaded encoder, decoder and joint onnx models !")

        # Will be populated at runtime
        self._blank_index = None
        self.max_symbols_per_step = max_symbols_per_step

        self._setup_encoder_input_output_keys()
        self._setup_decoder_joint_input_output_keys()
        self._setup_blank_index()

    def _setup_encoder_input_output_keys(self):
        self.encoder_inputs = list(self.encoder_model.graph.input)
        self.encoder_outputs = list(self.encoder_model.graph.output)

    def _setup_decoder_joint_input_output_keys(self):
        self.decoder_joint_inputs = list(self.decoder_joint_model.graph.input)
        self.decoder_joint_outputs = list(self.decoder_joint_model.graph.output)

    def _setup_blank_index(self):
        # ASSUME: Single input with no time length information
        dynamic_dim = 257
        shapes = self.encoder_inputs[0].type.tensor_type.shape.dim
        ip_shape = []
        for shape in shapes:
            if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                ip_shape.append(dynamic_dim)  # replace dynamic axes with constant
            else:
                ip_shape.append(int(shape.dim_value))

        enc_logits, encoded_length = self.run_encoder(
            audio_signal=torch.randn(*ip_shape), length=torch.randint(0, 1, size=(dynamic_dim,))
        )

        # prepare states
        states = self._get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = self.run_decoder_joint(enc_logits, None, None, *states)
        log_probs, lengths = joint_out

        self._blank_index = log_probs.shape[-1] - 1  # last token of vocab size is blank token
        logging.info(
            f"Enc-Dec-Joint step was evaluated, blank token id = {self._blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    def run_encoder(self, audio_signal, length):
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.cpu().numpy()

        ip = {
            self.encoder_inputs[0].name: audio_signal,
            self.encoder_inputs[1].name: length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        return enc_out, encoded_length

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = torch.zeros(enc_logits.shape[0], 1, dtype=torch.int32)
            target_length = torch.ones(enc_logits.shape[0], dtype=torch.int32)

        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        if hasattr(target_length, 'cpu'):
            target_length = target_length.cpu().numpy()

        ip = {
            self.decoder_joint_inputs[0].name: enc_logits,
            self.decoder_joint_inputs[1].name: targets,
            self.decoder_joint_inputs[2].name: target_length,
        }

        num_states = 0
        if states is not None and len(states) > 0:
            num_states = len(states)
            for idx, state in enumerate(states):
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()

                ip[self.decoder_joint_inputs[len(ip)].name] = state

        dec_out = self.decoder_joint.run(None, ip)

        # unpack dec output
        if num_states > 0:
            new_states = dec_out[-num_states:]
            dec_out = dec_out[:-num_states]
        else:
            new_states = None

        return dec_out, new_states

    def _get_initial_states(self, batchsize):
        # ASSUME: LSTM STATES of shape (layers, batchsize, dim)
        input_state_nodes = [ip for ip in self.decoder_joint_inputs if 'state' in ip.name]
        num_states = len(input_state_nodes)
        if num_states == 0:
            return

        input_states = []
        for state_id in range(num_states):
            node = input_state_nodes[state_id]
            ip_shape = []
            for shape_idx, shape in enumerate(node.type.tensor_type.shape.dim):
                if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                    ip_shape.append(batchsize)  # replace dynamic axes with constant
                else:
                    ip_shape.append(int(shape.dim_value))

            input_states.append(torch.zeros(*ip_shape))

        return input_states


decoding = ONNXGreedyBatchedRNNTInfer("stt_es_fastconformer_hybrid_large_pc/encoder.onnx", "stt_es_fastconformer_hybrid_large_pc/decoder_joint.onnx", max_symbols_per_step = 5)
audio_filepath = "/home/ubuntu/audio_dir/test.wav"


import soundfile as sf
import sentencepiece as spm
s = spm.SentencePieceProcessor(model_file='stt_es_fastconformer_hybrid_large_pc/tokenizer.model')

with sf.SoundFile(audio_filepath, 'r') as f:
    dtype = 'float32'
    sample_rate = f.samplerate
    samples = f.read(dtype=dtype)

input_signal = torch.tensor([np.array(samples)])
input_signal_length = torch.tensor([len(samples)])
            # input_signal = input_signal.to(device)
            # input_signal_length = input_signal_length.to("cuda")
processed_audio, processed_audio_len = preprocessor.get_features(
                 input_signal=input_signal, length=input_signal_length
             )
hypotheses = decoding(audio_signal=processed_audio, length=processed_audio_len)
prediction = hypotheses[0].y_sequence
prediction = prediction.tolist()
prediction = s.decode_ids(prediction)
print(prediction)


