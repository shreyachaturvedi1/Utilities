from abc import ABC
import torch
import abc
from abc import ABC
from typing import Dict, Optional, Tuple
from enum import Enum
from typing import Optional
import math
import random
from typing import Optional, Tuple, Union
import librosa
import numpy as np
import torch
import torch.nn as nn
import os
import random
from typing import Optional, Union
import math
import os
import random
from typing import Optional
import librosa
import numpy as np
import soundfile as sf
import librosa
import numpy as np
import soundfile as sf

from logger import get_logger
logger = get_logger(__name__)
try:
    import torchaudio

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


CONSTANT = 1e-5

HAVE_PYDUB = True
try:
    from pydub import AudioSegment as Audio
    from pydub.exceptions import CouldntDecodeError
except ModuleNotFoundError:
    HAVE_PYDUB = False


available_formats = sf.available_formats()
sf_supported_formats = ["." + i.lower() for i in available_formats.keys()]


class NeuralTypeComparisonResult(Enum):
    """The result of comparing two neural type objects for compatibility.
    When comparing A.compare_to(B):"""

    SAME = 0
    LESS = 1  # A is B
    GREATER = 2  # B is A
    DIM_INCOMPATIBLE = 3  # Resize connector might fix incompatibility
    TRANSPOSE_SAME = 4  # A transpose and/or converting between lists and tensors will make them same
    CONTAINER_SIZE_MISMATCH = 5  # A and B contain different number of elements
    INCOMPATIBLE = 6  # A and B are incompatible
    SAME_TYPE_INCOMPATIBLE_PARAMS = 7  # A and B are of the same type but parametrized differently
    UNCHECKED = 8  # type comparison wasn't done


class ElementType(ABC):
    """Abstract class defining semantics of the tensor elements.
    We are relying on Python for inheritance checking"""

    def __str__(self):
        return self.__doc__

    def __repr__(self):
        return self.__class__.__name__

    @property
    def type_parameters(self) -> Dict:
        """Override this property to parametrize your type. For example, you can specify 'storage' type such as
        float, int, bool with 'dtype' keyword. Another example, is if you want to represent a signal with a
        particular property (say, sample frequency), then you can put sample_freq->value in there.
        When two types are compared their type_parameters must match."""
        return {}

    @property
    def fields(self) -> Optional[Tuple]:
        """This should be used to logically represent tuples/structures. For example, if you want to represent a
        bounding box (x, y, width, height) you can put a tuple with names ('x', y', 'w', 'h') in here.
        Under the hood this should be converted to the last tesnor dimension of fixed size = len(fields).
        When two types are compared their fields must match."""
        return None

    def compare(self, second) -> NeuralTypeComparisonResult:
        # First, check general compatibility
        first_t = type(self)
        second_t = type(second)

        if first_t == second_t:
            result = NeuralTypeComparisonResult.SAME
        elif issubclass(first_t, second_t):
            result = NeuralTypeComparisonResult.LESS
        elif issubclass(second_t, first_t):
            result = NeuralTypeComparisonResult.GREATER
        else:
            result = NeuralTypeComparisonResult.INCOMPATIBLE

        if result != NeuralTypeComparisonResult.SAME:
            return result
        else:
            # now check that all parameters match
            check_params = set(self.type_parameters.keys()) == set(second.type_parameters.keys())
            if check_params is False:
                return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            else:
                for k1, v1 in self.type_parameters.items():
                    if v1 is None or second.type_parameters[k1] is None:
                        # Treat None as Void
                        continue
                    if v1 != second.type_parameters[k1]:
                        return NeuralTypeComparisonResult.SAME_TYPE_INCOMPATIBLE_PARAMS
            # check that all fields match
            if self.fields == second.fields:
                return NeuralTypeComparisonResult.SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE


class VoidType(ElementType):
    """Void-like type which is compatible with everything.
    It is a good practice to use this type only as necessary.
    For example, when you need template-like functionality.
    """

    def compare(cls, second: abc.ABCMeta) -> NeuralTypeComparisonResult:
        return NeuralTypeComparisonResult.SAME

class AxisKindAbstract(Enum):
    """This is an abstract Enum to represents what does varying axis dimension mean.
    In practice, you will almost always use AxisKind Enum. This Enum should be inherited by
    your OWN Enum if you aren't satisfied with AxisKind. Then your own Enum can be used
    instead of AxisKind."""

    pass


class AxisKind(AxisKindAbstract):
    """This Enum represents what does varying axis dimension mean.
    For example, does this dimension correspond to width, batch, time, etc.
    The "Dimension" and "Channel" kinds are the same and used to represent
    a general axis. "Any" axis will accept any axis kind fed to it.
    """

    Batch = 0
    Time = 1
    Dimension = 2
    Channel = 2
    Width = 3
    Height = 4
    Any = 5
    Sequence = 6
    FlowGroup = 7
    Singleton = 8  # Used to represent a axis that has size 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.name).lower()

    def t_with_string(self, text):
        # it checks if text is "t_<any string>"
        return text.startswith("t_") and text.endswith("_") and text[2:-1] == self.__str__()

    @staticmethod
    def from_str(label):
        """Returns AxisKind instance based on short string representation"""
        _label = label.lower().strip()
        if _label == "b" or _label == "n" or _label == "batch":
            return AxisKind.Batch
        elif _label == "t" or _label == "time" or (len(_label) > 2 and _label.startswith("t_")):
            return AxisKind.Time
        elif _label == "d" or _label == "c" or _label == "channel":
            return AxisKind.Dimension
        elif _label == "w" or _label == "width":
            return AxisKind.Width
        elif _label == "h" or _label == "height":
            return AxisKind.Height
        elif _label == "s" or _label == "singleton":
            return AxisKind.Singleton
        elif _label == "seq" or _label == "sequence":
            return AxisKind.Sequence
        elif _label == "flowgroup":
            return AxisKind.FlowGroup
        elif _label == "any":
            return AxisKind.Any
        else:
            raise ValueError(f"Can't create AxisKind from {label}")


class AxisType(object):
    """This class represents axis semantics and (optionally) it's dimensionality
       Args:
           kind (AxisKindAbstract): what kind of axis it is? For example Batch, Height, etc.
           size (int, optional): specify if the axis should have a fixed size. By default it is set to None and you
           typically do not want to set it for Batch and Time
           is_list (bool, default=False): whether this is a list or a tensor axis
    """

    def __init__(self, kind: AxisKindAbstract, size: Optional[int] = None, is_list=False):
        if size is not None and is_list:
            raise ValueError("The axis can't be list and have a fixed size")
        self.kind = kind
        self.size = size
        self.is_list = is_list

    def __repr__(self):
        if self.size is None:
            representation = str(self.kind)
        else:
            representation = f"{str(self.kind)}:{self.size}"
        if self.is_list:
            representation += "_listdim"
        return representation


class NeuralType(object):
    """This is the main class which would represent neural type concept.
    It is used to represent *the types* of inputs and outputs.

    Args:
        axes (Optional[Tuple]): a tuple of AxisTypes objects representing the semantics of what varying each axis means
            You can use a short, string-based form here. For example: ('B', 'C', 'H', 'W') would correspond to an NCHW
            format frequently used in computer vision. ('B', 'T', 'D') is frequently used for signal processing and
            means [batch, time, dimension/channel].
        elements_type (ElementType): an instance of ElementType class representing the semantics of what is stored
            inside the tensor. For example: logits (LogitsType), log probabilities (LogprobType), etc.
        optional (bool): By default, this is false. If set to True, it would means that input to the port of this
            type can be optional.
    """

    def __str__(self):

        if self.axes is not None:
            return f"axes: {self.axes}; elements_type: {self.elements_type.__class__.__name__}"
        else:
            return f"axes: None; elements_type: {self.elements_type.__class__.__name__}"

    def __init__(self, axes: Optional[Tuple] = None, elements_type: ElementType = VoidType(), optional=False):
        if not isinstance(elements_type, ElementType):
            raise ValueError(
                "elements_type of NeuralType must be an instance of a class derived from ElementType. "
                "Did you pass a class instead?"
            )
        self.elements_type = elements_type
        if axes is not None:
            NeuralType.__check_sanity(axes)
            axes_list = []
            for axis in axes:
                if isinstance(axis, str):
                    axes_list.append(AxisType(AxisKind.from_str(axis), None))
                elif isinstance(axis, AxisType):
                    axes_list.append(axis)
                else:
                    raise ValueError("axis type must be either str or AxisType instance")
            self.axes = tuple(axes_list)
        else:
            self.axes = None
        self.optional = optional

    def compare(self, second) -> NeuralTypeComparisonResult:
        """Performs neural type comparison of self with second. When you chain two modules' inputs/outputs via
        __call__ method, this comparison will be called to ensure neural type compatibility."""
        # First, handle dimensionality
        axes_a = self.axes
        axes_b = second.axes

        # "Big void" type
        if isinstance(self.elements_type, VoidType) and self.axes is None:
            return NeuralTypeComparisonResult.SAME

        if self.axes is None:
            if second.axes is None:
                return self.elements_type.compare(second.elements_type)
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE

        dimensions_pass = NeuralType.__compare_axes(axes_a, axes_b)
        element_comparison_result = self.elements_type.compare(second.elements_type)

        # SAME DIMS
        if dimensions_pass == 0:
            return element_comparison_result
        # TRANSPOSE_SAME DIMS
        elif dimensions_pass == 1:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.TRANSPOSE_SAME
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        # DIM_INCOMPATIBLE DIMS
        elif dimensions_pass == 2:
            if element_comparison_result == NeuralTypeComparisonResult.SAME:
                return NeuralTypeComparisonResult.DIM_INCOMPATIBLE
            else:
                return NeuralTypeComparisonResult.INCOMPATIBLE
        else:
            return NeuralTypeComparisonResult.INCOMPATIBLE

    def compare_and_raise_error(self, parent_type_name, port_name, second_object):
        """ Method compares definition of one type with another and raises an error if not compatible. """
        type_comatibility = self.compare(second_object)
        if (
            type_comatibility != NeuralTypeComparisonResult.SAME
            and type_comatibility != NeuralTypeComparisonResult.GREATER
        ):
            raise NeuralPortNmTensorMismatchError(
                parent_type_name, port_name, str(self), str(second_object.ntype), type_comatibility
            )

    def __eq__(self, other):
        if isinstance(other, NeuralType):
            return self.compare(other)

        return False

    @staticmethod
    def __check_sanity(axes):
        # check that list come before any tensor dimension
        are_strings = True
        for axis in axes:
            if not isinstance(axis, str):
                are_strings = False
            if isinstance(axis, str) and not are_strings:
                raise ValueError("Either use full class names or all strings")
        if are_strings:
            return
        checks_passed = True
        saw_tensor_dim = False
        for axis in axes:
            if not axis.is_list:
                saw_tensor_dim = True
            else:  # current axis is a list
                if saw_tensor_dim:  # which is preceded by tensor dim
                    checks_passed = False
        if not checks_passed:
            raise ValueError(
                "You have list dimension after Tensor dimension. All list dimensions must preceed Tensor dimensions"
            )

    @staticmethod
    def __compare_axes(axes_a, axes_b) -> int:
        """
        Compares axes_a and axes_b
        Args:
            axes_a: first axes tuple
            axes_b: second axes tuple

        Returns:
            0 - if they are exactly the same
            1 - if they are "TRANSPOSE_SAME"
            2 - if the are "DIM_INCOMPATIBLE"
            3 - if they are different
        """
        if axes_a is None and axes_b is None:
            return 0
        elif axes_a is None and axes_b is not None:
            return 3
        elif axes_a is not None and axes_b is None:
            return 3
        elif len(axes_a) != len(axes_b):
            return 3
        # After these ifs we know that len(axes_a) == len(axes_b)

        same = True
        kinds_a = dict()
        kinds_b = dict()
        for axis_a, axis_b in zip(axes_a, axes_b):
            kinds_a[axis_a.kind] = axis_a.size
            kinds_b[axis_b.kind] = axis_b.size
            if axis_a.kind == AxisKind.Any:
                same = True
            elif (
                axis_a.kind != axis_b.kind
                or axis_a.is_list != axis_b.is_list
                or (axis_a.size != axis_b.size and axis_a.size is not None)
            ):
                same = False
        if same:
            return 0
        else:
            # can be TRANSPOSE_SAME, DIM_INCOMPATIBLE
            if kinds_a.keys() == kinds_b.keys():
                for key, value in kinds_a.items():
                    if kinds_b[key] != value:
                        return 2
                return 1
            else:
                return 3

    def __repr__(self):
        if self.axes is not None:
            axes = str(self.axes)
        else:
            axes = "None"

        if self.elements_type is not None:
            element_type = repr(self.elements_type)
        else:
            element_type = "None"

        data = f"axis={axes}, element_type={element_type}"

        if self.optional:
            data = f"{data}, optional={self.optional}"

        final = f"{self.__class__.__name__}({data})"
        return final


class NeuralTypeError(Exception):
    """Base class for neural type related exceptions."""



class LengthsType(ElementType):
    """Element type representing lengths of something"""

class AudioSignal(ElementType):
    """Element type to represent encoded representation returned by the acoustic encoder model
    Args:
        freq (int): sampling frequency of a signal. Note that two signals will only be the same if their
        freq is the same.
    """

    def __init__(self, freq: int = None):
        self._params = {}
        self._params['freq'] = freq

    @property
    def type_parameters(self):
        return self._params

class ChannelType(ElementType):
    """Element to represent convolutional input/output channel.
    """

class SpectrogramType(ChannelType):
    """Element type to represent generic spectrogram signal"""

class MelSpectrogramType(SpectrogramType):
    """Element type to represent mel spectrogram signal"""


class Perturbation(object):
    def max_augmentation_length(self, length):
        return length

    def perturb(self, data):
        raise NotImplementedError

class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        random.seed(rng) if rng else None
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if random.random() < prob:
                p.perturb(segment)
        return

    def max_augmentation_length(self, length):
        newlen = length
        for (prob, p) in self._pipeline:
            newlen = p.max_augmentation_length(newlen)
        return newlen

    @classmethod
    def from_config(cls, config):
        ptbs = []
        return cls(perturbations=ptbs)


class AudioSegment(object):
    """Audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(
        self,
        samples,
        sample_rate,
        target_sr=None,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db: Optional[float] = None,
        ref_channel: Optional[int] = None,
    ):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)

        # Check if channel selector is necessary
        if samples.ndim == 1 and channel_selector not in [None, 0, 'average']:
            raise ValueError(
                'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
            )

        if target_sr is not None and target_sr != sample_rate:
            # resample along the temporal dimension (axis=0) will be in librosa 0.10.0 (#1561)
            samples = samples.transpose()
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
            samples = samples.transpose()
            sample_rate = target_sr
        if trim:
            # librosa is using channels-first layout (num_channels, num_samples), which is transpose of AudioSegment's layout
            samples = samples.transpose()
            samples, _ = librosa.effects.trim(
                samples, top_db=trim_top_db, ref=trim_ref, frame_length=trim_frame_length, hop_length=trim_hop_length
            )
            samples = samples.transpose()
        self._samples = samples
        self._sample_rate = sample_rate
        self._orig_sr = orig_sr if orig_sr is not None else sample_rate
        self._ref_channel = ref_channel
        self._normalize_db = normalize_db

        if normalize_db is not None:
            self.normalize_db(normalize_db, ref_channel)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        if self.num_channels == 1:
            return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, rms=%.2fdB" % (
                type(self),
                self.num_samples,
                self.sample_rate,
                self.duration,
                self.rms_db,
            )
        else:
            rms_db_str = ', '.join([f'{rms:.2f}dB' for rms in self.rms_db])
            return "%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, num_channels=%d, rms=[%s]" % (
                type(self),
                self.num_samples,
                self.sample_rate,
                self.duration,
                self.num_channels,
                rms_db_str,
            )

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(
        cls,
        audio_file,
        target_sr=None,
        int_values=False,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
        ref_channel=None,
    ):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param audio_file: path of file to load.
                           Alternatively, a list of paths of single-channel files can be provided
                           to form a multichannel signal.
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :param trim: if true, trim leading and trailing silence from an audio signal
        :param trim_ref: the reference amplitude. By default, it uses `np.max` and compares to the peak amplitude in
                         the signal
        :param trim_top_db: the threshold (in decibels) below reference to consider as silence
        :param trim_frame_length: the number of samples per analysis frame
        :param trim_hop_length: the number of samples between analysis frames
        :param orig_sr: the original sample rate
        :param channel selector: string denoting the downmix mode, an integer denoting the channel to be selected, or an iterable
                                 of integers denoting a subset of channels. Channel selector is using zero-based indexing.
                                 If set to `None`, the original signal will be used.
        :param normalize_db (Optional[float]): if not None, normalize the audio signal to a target RMS value
        :param ref_channel (Optional[int]): channel to use as reference for normalizing multi-channel audio, set None to use max RMS across channels
        :return: AudioSegment instance
        """
        samples = None
        if isinstance(audio_file, list):
            return cls.from_file_list(
                audio_file_list=audio_file,
                target_sr=target_sr,
                int_values=int_values,
                offset=offset,
                duration=duration,
                trim=trim,
                trim_ref=trim_ref,
                trim_top_db=trim_top_db,
                trim_frame_length=trim_frame_length,
                trim_hop_length=trim_hop_length,
                orig_sr=orig_sr,
                channel_selector=channel_selector,
                normalize_db=normalize_db,
                ref_channel=ref_channel,
            )

        if not isinstance(audio_file, str) or os.path.splitext(audio_file)[-1] in sf_supported_formats:
            try:
                with sf.SoundFile(audio_file, 'r') as f:
                    dtype = 'int32' if int_values else 'float32'
                    sample_rate = f.samplerate
                    if offset > 0:
                        f.seek(int(offset * sample_rate))
                    if duration > 0:
                        samples = f.read(int(duration * sample_rate), dtype=dtype)
                    else:
                        samples = f.read(dtype=dtype)
            except RuntimeError as e:
                logger.error(
                    f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`. "
                    f"NeMo will fallback to loading via pydub."
                )

                if hasattr(audio_file, "seek"):
                    audio_file.seek(0)

        if HAVE_PYDUB and samples is None:
            try:
                samples = Audio.from_file(audio_file)
                sample_rate = samples.frame_rate
                num_channels = samples.channels
                if offset > 0:
                    # pydub does things in milliseconds
                    seconds = offset * 1000
                    samples = samples[int(seconds) :]
                if duration > 0:
                    seconds = duration * 1000
                    samples = samples[: int(seconds)]
                samples = np.array(samples.get_array_of_samples())
                # For multi-channel signals, channels are stacked in a one-dimensional vector
                if num_channels > 1:
                    samples = np.reshape(samples, (-1, num_channels))
            except CouldntDecodeError as err:
                logger.error(f"Loading {audio_file} via pydub raised CouldntDecodeError: `{err}`.")

        if samples is None:
            libs = "soundfile, and pydub" if HAVE_PYDUB else "soundfile"
            raise Exception(f"Your audio file {audio_file} could not be decoded. We tried using {libs}.")

        return cls(
            samples,
            sample_rate,
            target_sr=target_sr,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
            ref_channel=ref_channel,
        )

    @classmethod
    def from_file_list(
        cls,
        audio_file_list,
        target_sr=None,
        int_values=False,
        offset=0,
        duration=0,
        trim=False,
        channel_selector=None,
        *args,
        **kwargs,
    ):
        """
        Function wrapper for `from_file` method. Load a list of files from `audio_file_list`.
        The length of each audio file is unified with the duration item in the input manifest file.
        See `from_file` method for arguments.

        If a list of files is provided, load samples from individual single-channel files and
        concatenate them along the channel dimension.
        """
        if isinstance(channel_selector, int):
            # Shortcut when selecting a single channel
            if channel_selector >= len(audio_file_list):
                raise RuntimeError(
                    f'Channel cannot be selected: channel_selector={channel_selector}, num_audio_files={len(audio_file_list)}'
                )
            # Select only a single file
            audio_file_list = [audio_file_list[channel_selector]]
            # Reset the channel selector since we applied it here
            channel_selector = None

        samples = None

        for a_file in audio_file_list:
            # Load audio from the current file
            a_segment = cls.from_file(
                a_file,
                target_sr=target_sr,
                int_values=int_values,
                offset=offset,
                duration=duration,
                channel_selector=None,
                trim=False,  # Do not apply trim to individual files, it will be applied to the concatenated signal
                *args,
                **kwargs,
            )

            # Only single-channel individual files are supported for now
            if a_segment.num_channels != 1:
                raise RuntimeError(
                    f'Expecting a single-channel audio signal, but loaded {a_segment.num_channels} channels from file {a_file}'
                )

            if target_sr is None:
                # All files need to be loaded with the same sample rate
                target_sr = a_segment.sample_rate

            # Concatenate samples
            a_samples = a_segment.samples[:, None]

            if samples is None:
                samples = a_samples
            else:
                # Check the dimensions match
                if len(a_samples) != len(samples):
                    raise RuntimeError(
                        f'Loaded samples need to have identical length: {a_samples.shape} != {samples.shape}'
                    )

                # Concatenate along channel dimension
                samples = np.concatenate([samples, a_samples], axis=1)

        # Final setup for class initialization
        samples = np.squeeze(samples)
        sample_rate = target_sr

        return cls(
            samples, sample_rate, target_sr=target_sr, trim=trim, channel_selector=channel_selector, *args, **kwargs,
        )

    @classmethod
    def segment_from_file(
        cls, audio_file, target_sr=None, n_segments=0, trim=False, orig_sr=None, channel_selector=None, offset=None
    ):
        """Grabs n_segments number of samples from audio_file.
        If offset is not provided, n_segments are selected randomly.
        If offset is provided, it is used to calculate the starting sample.

        Note that audio_file can be either the file path, or a file-like object.

        :param audio_file: path to a file or a file-like object
        :param target_sr: sample rate for the output samples
        :param n_segments: desired number of samples
        :param trim: if true, trim leading and trailing silence from an audio signal
        :param orig_sr: the original sample rate
        :param channel selector: select a subset of channels. If set to `None`, the original signal will be used.
        :param offset: fixed offset in seconds
        :return: numpy array of samples
        """
        is_segmented = False
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                sample_rate = f.samplerate
                if target_sr is not None:
                    n_segments_at_original_sr = math.ceil(n_segments * sample_rate / target_sr)
                else:
                    n_segments_at_original_sr = n_segments

                if 0 < n_segments_at_original_sr < len(f):
                    max_audio_start = len(f) - n_segments_at_original_sr
                    if offset is None:
                        audio_start = random.randint(0, max_audio_start)
                    else:
                        audio_start = math.floor(offset * sample_rate)
                        if audio_start > max_audio_start:
                            raise RuntimeError(
                                f'Provided audio start ({audio_start}) is larger than the maximum possible ({max_audio_start})'
                            )
                    f.seek(audio_start)
                    samples = f.read(n_segments_at_original_sr, dtype='float32')
                    is_segmented = True
                elif n_segments_at_original_sr > len(f):
                    logger.warning(
                        f"Number of segments ({n_segments_at_original_sr}) is greater than the length ({len(f)}) of the audio file {audio_file}. This may lead to shape mismatch errors."
                    )
                    samples = f.read(dtype='float32')
                else:
                    samples = f.read(dtype='float32')
        except RuntimeError as e:
            logger.error(f"Loading {audio_file} via SoundFile raised RuntimeError: `{e}`.")
            raise e

        features = cls(
            samples, sample_rate, target_sr=target_sr, trim=trim, orig_sr=orig_sr, channel_selector=channel_selector
        )

        if is_segmented:
            features._samples = features._samples[:n_segments]

        return features

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_channels(self):
        if self._samples.ndim == 1:
            return 1
        else:
            return self._samples.shape[-1]

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self.num_samples / float(self._sample_rate)

    @property
    def rms_db(self):
        """Return per-channel RMS value.
        """
        mean_square = np.mean(self._samples ** 2, axis=0)
        return 10 * np.log10(mean_square)

    @property
    def orig_sr(self):
        return self._orig_sr

    def gain_db(self, gain):
        self._samples *= 10.0 ** (gain / 20.0)

    def normalize_db(self, target_db=-20, ref_channel=None):
        """Normalize the signal to a target RMS value in decibels. 
        For multi-channel audio, the RMS value is determined by the reference channel (if not None),
        otherwise it will be the maximum RMS across all channels.
        """
        rms_db = self.rms_db
        if self.num_channels > 1:
            rms_db = max(rms_db) if ref_channel is None else rms_db[ref_channel]
        gain = target_db - rms_db
        self.gain_db(gain)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number
        of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false,
        `pad_size`
        zeros will be added only to the end.
        """
        samples_ndim = self._samples.ndim
        if samples_ndim == 1:
            pad_width = pad_size if symmetric else (0, pad_size)
        elif samples_ndim == 2:
            # pad samples, keep channels
            pad_width = ((pad_size, pad_size), (0, 0)) if symmetric else ((0, pad_size), (0, 0))
        else:
            raise NotImplementedError(
                f"Padding not implemented for signals with more that 2 dimensions. Current samples dimension: {samples_ndim}."
            )
        # apply padding
        self._samples = np.pad(self._samples, pad_width, mode='constant',)

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set,
        e.g. out of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time
        if start_time < 0.0:
            start_time = self.duration + start_time
        if end_time < 0.0:
            end_time = self.duration + end_time
        if start_time < 0.0:
            raise ValueError("The slice start position (%f s) is out of bounds." % start_time)
        if end_time < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." % end_time)
        if start_time > end_time:
            raise ValueError(
                "The slice start position (%f s) is later than the end position (%f s)." % (start_time, end_time)
            )
        if end_time > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds (> %f s)" % (end_time, self.duration))
        start_sample = int(round(start_time * self._sample_rate))
        end_sample = int(round(end_time * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                    "feature (ex. at least `hop_length` for Mel Spectrograms)."
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


def clean_spectrogram_batch(spectrogram: torch.Tensor, spectrogram_len: torch.Tensor, fill_value=0.0) -> torch.Tensor:
    """
    Fill spectrogram values outside the length with `fill_value`

    Args:
        spectrogram: Tensor with shape [B, C, L] containing batched spectrograms
        spectrogram_len: Tensor with shape [B] containing the sequence length of each batch element
        fill_value: value to fill with, 0.0 by default

    Returns:
        cleaned spectrogram, tensor with shape equal to `spectrogram`
    """
    device = spectrogram.device
    batch_size, _, max_len = spectrogram.shape
    mask = torch.arange(max_len, device=device)[None, :] >= spectrogram_len[:, None]
    mask = mask.unsqueeze(1).expand_as(spectrogram)
    return spectrogram.masked_fill(mask, fill_value)


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


@torch.jit.script_if_tracing
def make_seq_mask_like(
    lengths: torch.Tensor, like: torch.Tensor, time_dim: int = -1, valid_ones: bool = True
) -> torch.Tensor:
    """

    Args:
        lengths: Tensor with shape [B] containing the sequence length of each batch element
        like: The mask will contain the same number of dimensions as this Tensor, and will have the same max
            length in the time dimension of this Tensor.
        time_dim: Time dimension of the `shape_tensor` and the resulting mask. Zero-based.
        valid_ones: If True, valid tokens will contain value `1` and padding will be `0`. Else, invert.

    Returns:
        A :class:`torch.Tensor` containing 1's and 0's for valid and invalid tokens, respectively, if `valid_ones`, else
        vice-versa. Mask will have the same number of dimensions as `like`. Batch and time dimensions will match
        the `like`. All other dimensions will be singletons. E.g., if `like.shape == [3, 4, 5]` and
        `time_dim == -1', mask will have shape `[3, 1, 5]`.
    """
    # Mask with shape [B, T]
    mask = torch.arange(like.shape[time_dim], device=like.device).repeat(lengths.shape[0], 1).lt(lengths.view(-1, 1))
    # [B, T] -> [B, *, T] where * is any number of singleton dimensions to expand to like tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If needed, transpose time dim
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(-1, time_dim)
    # Maybe invert the padded vs. valid token values
    if not valid_ones:
        mask = ~mask
    return mask


class WaveformFeaturizer(object):
    def __init__(self, sample_rate=16000, int_values=False, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.sample_rate = sample_rate
        self.int_values = int_values

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(
        self,
        file_path,
        offset=0,
        duration=0,
        trim=False,
        trim_ref=np.max,
        trim_top_db=60,
        trim_frame_length=2048,
        trim_hop_length=512,
        orig_sr=None,
        channel_selector=None,
        normalize_db=None,
    ):
        audio = AudioSegment.from_file(
            file_path,
            target_sr=self.sample_rate,
            int_values=self.int_values,
            offset=offset,
            duration=duration,
            trim=trim,
            trim_ref=trim_ref,
            trim_top_db=trim_top_db,
            trim_frame_length=trim_frame_length,
            trim_hop_length=trim_hop_length,
            orig_sr=orig_sr,
            channel_selector=channel_selector,
            normalize_db=normalize_db,
        )
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        sample_rate = input_config.get("sample_rate", 16000)
        int_values = input_config.get("int_values", False)

        return cls(sample_rate=sample_rate, int_values=int_values, augmentor=aa)


class FeaturizerFactory(object):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, input_cfg, perturbation_configs=None):
        return WaveformFeaturizer.from_config(input_cfg, perturbation_configs=perturbation_configs)


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logger.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logger.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None

        if exact_pad:
            logger.info("STFT using exact pad")
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logger.debug(f"sr: {sample_rate}")
        logger.debug(f"n_fft: {self.n_fft}")
        logger.debug(f"win_length: {self.win_length}")
        logger.debug(f"hop_length: {self.hop_length}")
        logger.debug(f"n_mels: {nfilt}")
        logger.debug(f"fmin: {lowfreq}")
        logger.debug(f"fmax: {highfreq}")
        logger.debug(f"using grads: {use_grads}")
        logger.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor((seq_len + pad_amount - self.n_fft) / self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len.float())

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len


class FilterbankFeaturesTA(nn.Module):
    """
    Exportable, `torchaudio`-based implementation of Mel Spectrogram extraction.

    See `AudioToMelSpectrogramPreprocessor` for args.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        normalize: Optional[str] = "per_feature",
        nfilt: int = 64,
        n_fft: Optional[int] = None,
        preemph: float = 0.97,
        lowfreq: float = 0,
        highfreq: Optional[float] = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: Union[float, str] = 2 ** -24,
        dither: float = 1e-5,
        window: str = "hann",
        pad_to: int = 0,
        pad_value: float = 0.0,
        mel_norm="slaney",
        # Seems like no one uses these options anymore. Don't convolute the code by supporting thm.
        use_grads: bool = False,  # Deprecated arguments; kept for config compatibility
        max_duration: float = 16.7,  # Deprecated arguments; kept for config compatibility
        frame_splicing: int = 1,  # Deprecated arguments; kept for config compatibility
        exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        nb_augmentation_prob: float = 0.0,  # Deprecated arguments; kept for config compatibility
        nb_max_freq: int = 4000,  # Deprecated arguments; kept for config compatibility
        mag_power: float = 2.0,  # Deprecated arguments; kept for config compatibility
        rng: Optional[random.Random] = None,  # Deprecated arguments; kept for config compatibility
        stft_exact_pad: bool = False,  # Deprecated arguments; kept for config compatibility
        stft_conv: bool = False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if not HAVE_TORCHAUDIO:
            raise ValueError(f"Need to install torchaudio to instantiate a {self.__class__.__name__}")

        # Make sure log zero guard is supported, if given as a string
        supported_log_zero_guard_strings = {"eps", "tiny"}
        if isinstance(log_zero_guard_value, str) and log_zero_guard_value not in supported_log_zero_guard_strings:
            raise ValueError(
                f"Log zero guard value must either be a float or a member of {supported_log_zero_guard_strings}"
            )

        # Copied from `AudioPreprocessor` due to the ad-hoc structuring of the Mel Spec extractor class
        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

        # Ensure we can look up the window function
        if window not in self.torch_windows:
            raise ValueError(f"Got window value '{window}' but expected a member of {self.torch_windows.keys()}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self._sample_rate = sample_rate
        self._normalize_strategy = normalize
        self._use_log = log
        self._preemphasis_value = preemph
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value: Union[str, float] = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.n_fft = n_fft
        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=nfilt,
            window_fn=self.torch_windows[window],
            mel_scale="slaney",
            norm=mel_norm,
            n_fft=n_fft,
            f_max=highfreq,
            f_min=lowfreq,
            wkwargs={"periodic": False},
        )

    @property
    def filter_banks(self):
        """ Matches the analogous class """
        return self._mel_spec_extractor.mel_scale.fb

    def _resolve_log_zero_guard_value(self, dtype: torch.dtype) -> float:
        if isinstance(self.log_zero_guard_value, float):
            return self.log_zero_guard_value
        return getattr(torch.finfo(dtype), self.log_zero_guard_value)

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(self, signals: torch.Tensor) -> torch.Tensor:
        if self._preemphasis_value is not None:
            padded = torch.nn.functional.pad(signals, (1, 0))
            signals = signals - self._preemphasis_value * padded[:, :-1]
        return signals

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        out_lengths = input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
        return out_lengths

    def _apply_pad_to(self, features: torch.Tensor) -> torch.Tensor:
        # Only apply during training; else need to capture dynamic shape for exported models
        if not self.training or self.pad_to == 0 or features.shape[-1] % self.pad_to == 0:
            return features
        pad_length = self.pad_to - (features.shape[-1] % self.pad_to)
        return torch.nn.functional.pad(features, pad=(0, pad_length), value=self.pad_value)

    def _apply_log(self, features: torch.Tensor) -> torch.Tensor:
        if self._use_log:
            zero_guard = self._resolve_log_zero_guard_value(features.dtype)
            if self.log_zero_guard_type == "add":
                features = features + zero_guard
            elif self.log_zero_guard_type == "clamp":
                features = features.clamp(min=zero_guard)
            else:
                raise ValueError(f"Unsupported log zero guard type: '{self.log_zero_guard_type}'")
            features = features.log()
        return features

    def _extract_spectrograms(self, signals: torch.Tensor) -> torch.Tensor:
        # Complex FFT needs to be done in single precision
        with torch.cuda.amp.autocast(enabled=False):
            features = self._mel_spec_extractor(waveform=signals)
        return features

    def _apply_normalization(self, features: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # For consistency, this function always does a masked fill even if not normalizing.
        mask: torch.Tensor = make_seq_mask_like(lengths=lengths, like=features, time_dim=-1, valid_ones=False)
        features = features.masked_fill(mask, 0.0)
        # Maybe don't normalize
        if self._normalize_strategy is None:
            return features
        # Use the log zero guard for the sqrt zero guard
        guard_value = self._resolve_log_zero_guard_value(features.dtype)
        if self._normalize_strategy == "per_feature" or self._normalize_strategy == "all_features":
            # 'all_features' reduces over each sample; 'per_feature' reduces over each channel
            reduce_dim = 2
            if self._normalize_strategy == "all_features":
                reduce_dim = [1, 2]
            # [B, D, T] -> [B, D, 1] or [B, 1, 1]
            means = features.sum(dim=reduce_dim, keepdim=True).div(lengths.view(-1, 1, 1))
            stds = (
                features.sub(means)
                .masked_fill(mask, 0.0)
                .pow(2.0)
                .sum(dim=reduce_dim, keepdim=True)  # [B, D, T] -> [B, D, 1] or [B, 1, 1]
                .div(lengths.view(-1, 1, 1) - 1)  # assume biased estimator
                .clamp(min=guard_value)  # avoid sqrt(0)
                .sqrt()
            )
            features = (features - means) / (stds + eps)
        else:
            # Deprecating constant std/mean
            raise ValueError(f"Unsupported norm type: '{self._normalize_strategy}")
        features = features.masked_fill(mask, 0.0)
        return features

    def forward(self, input_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_lengths = self._compute_output_lengths(input_lengths=length)
        signals = self._apply_dithering(signals=input_signal)
        signals = self._apply_preemphasis(signals=signals)
        features = self._extract_spectrograms(signals=signals)
        features = self._apply_log(features=features)
        features = self._apply_normalization(features=features, lengths=feature_lengths)
        features = self._apply_pad_to(features=features)
        return features, feature_lengths

class AudioToMelSpectrogramPreprocessor(ABC):
    """Featurizer module that converts wavs to mel spectrograms.

        Args:
            sample_rate (int): Sample rate of the input audio data.
                Defaults to 16000
            window_size (float): Size of window for fft in seconds
                Defaults to 0.02
            window_stride (float): Stride of window for fft in seconds
                Defaults to 0.01
            n_window_size (int): Size of window for fft in samples
                Defaults to None. Use one of window_size or n_window_size.
            n_window_stride (int): Stride of window for fft in samples
                Defaults to None. Use one of window_stride or n_window_stride.
            window (str): Windowing function for fft. can be one of ['hann',
                'hamming', 'blackman', 'bartlett']
                Defaults to "hann"
            normalize (str): Can be one of ['per_feature', 'all_features']; all
                other options disable feature normalization. 'all_features'
                normalizes the entire spectrogram to be mean 0 with std 1.
                'pre_features' normalizes per channel / freq instead.
                Defaults to "per_feature"
            n_fft (int): Length of FT window. If None, it uses the smallest power
                of 2 that is larger than n_window_size.
                Defaults to None
            preemph (float): Amount of pre emphasis to add to audio. Can be
                disabled by passing None.
                Defaults to 0.97
            features (int): Number of mel spectrogram freq bins to output.
                Defaults to 64
            lowfreq (int): Lower bound on mel basis in Hz.
                Defaults to 0
            highfreq  (int): Lower bound on mel basis in Hz.
                Defaults to None
            log (bool): Log features.
                Defaults to True
            log_zero_guard_type(str): Need to avoid taking the log of zero. There
                are two options: "add" or "clamp".
                Defaults to "add".
            log_zero_guard_value(float, or str): Add or clamp requires the number
                to add with or clamp to. log_zero_guard_value can either be a float
                or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
                passed.
                Defaults to 2**-24.
            dither (float): Amount of white-noise dithering.
                Defaults to 1e-5
            pad_to (int): Ensures that the output size of the time dimension is
                a multiple of pad_to.
                Defaults to 16
            frame_splicing (int): Defaults to 1
            exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
                // hop_length. Defaults to False.
            pad_value (float): The value that shorter mels are padded with.
                Defaults to 0
            mag_power (float): The power that the linear spectrogram is raised to
                prior to multiplication with mel basis.
                Defaults to 2 for a power spec
            rng : Random number generator
            nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
                samples in the batch.
                Defaults to 0.0
            nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
                Defaults to 4000
            use_torchaudio: Whether to use the `torchaudio` implementation.
            mel_norm: Normalization used for mel filterbank weights.
                Defaults to 'slaney' (area normalization)
            stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
            stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
        """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),  # Please note that length should be in samples not seconds.
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.

        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(MelSpectrogramSignalTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        use_torchaudio: bool = False,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }
        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        if not use_torchaudio:
            featurizer_class = FilterbankFeatures
        else:
            featurizer_class = FilterbankFeaturesTA
        self.featurizer = featurizer_class(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,  # Deprecated arguments; kept for config compatibility
            stft_conv=stft_conv,  # Deprecated arguments; kept for config compatibility
        )

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks



