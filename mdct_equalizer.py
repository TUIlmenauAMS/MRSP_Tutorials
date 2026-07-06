#!/usr/bin/env python3
"""
Codex: Can you make a Python program for an audio equalizer, using the MDCT, with 8 or 16 subbands, with sliders for the attenuation or gain in each subband? It should use either the live microphone signal, or an audio file as input.

Gerald Schuller, June 2026 

Real-time MDCT audio equalizer with 8 or 16 subbands.

Examples:
    python3 mdct_equalizer.py --bands 16 --input mic
    python3 mdct_equalizer.py --bands 8 --input 04_topchart.wav
    python3 mdct_equalizer.py --list-devices
    python3 mdct_equalizer.py --input-device 3 --output-device 5

Dependencies:
    numpy, scipy, pyaudio
"""

from __future__ import annotations

import argparse
import queue
import struct
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pyaudio
import scipy.fftpack as spfft
import scipy.io.wavfile as wav


class MDCTFilterBank:
    """MDCT analysis/synthesis filter bank using the lecture D(z)/F/DCT4 form."""

    def __init__(self, subbands: int):
        self.n = subbands
        self._analysis_delay = np.zeros(subbands // 2)
        self._synthesis_delay = np.zeros(subbands // 2)

        fcoeff = np.sin(np.pi / (2 * subbands) * (np.arange(0, 2 * subbands) + 0.5))
        fmatrix = np.zeros((subbands, subbands))
        h = subbands // 2
        fmatrix[0:h, 0:h] = np.fliplr(np.diag(fcoeff[0:h]))
        fmatrix[h:subbands, 0:h] = np.diag(fcoeff[h:subbands])
        fmatrix[0:h, h:subbands] = np.diag(fcoeff[subbands : subbands + h])
        fmatrix[h:subbands, h:subbands] = -np.fliplr(
            np.diag(fcoeff[subbands + h : 2 * subbands])
        )

        self._fmatrix = fmatrix
        self._finv = np.linalg.inv(fmatrix)

    def reset(self) -> None:
        self._analysis_delay[:] = 0.0
        self._synthesis_delay[:] = 0.0

    def _dct4(self, samples: np.ndarray) -> np.ndarray:
        samples_up = np.zeros(2 * self.n)
        samples_up[1::2] = samples
        return spfft.dct(samples_up, type=3)[: self.n] / 2.0

    def _analysis_dmatrix(self, samples: np.ndarray) -> np.ndarray:
        h = self.n // 2
        out = np.zeros(self.n)
        out[0:h] = self._analysis_delay
        self._analysis_delay = samples[0:h].copy()
        out[h : self.n] = samples[h : self.n]
        return out

    def _synthesis_dmatrix(self, samples: np.ndarray) -> np.ndarray:
        h = self.n // 2
        out = np.zeros(self.n)
        out[h : self.n] = self._synthesis_delay
        self._synthesis_delay = samples[h : self.n].copy()
        out[0:h] = samples[0:h]
        return out

    def analysis(self, samples: np.ndarray) -> np.ndarray:
        y = np.dot(samples, self._fmatrix)
        y = self._analysis_dmatrix(y)
        return self._dct4(y)

    def synthesis(self, coeffs: np.ndarray) -> np.ndarray:
        x = self._dct4(coeffs) * 2.0 / self.n
        x = self._synthesis_dmatrix(x)
        return np.dot(x, self._finv)

    def process_block(self, samples: np.ndarray, gains: np.ndarray) -> np.ndarray:
        return self.synthesis(self.analysis(samples) * gains)


class SharedGains:
    def __init__(self, bands: int):
        self._lock = threading.Lock()
        self._db = np.zeros(bands, dtype=np.float64)
        self._linear = np.ones(bands, dtype=np.float64)

    def set_db(self, band: int, value_db: float) -> None:
        with self._lock:
            self._db[band] = value_db
            self._linear[band] = 10.0 ** (value_db / 20.0)

    def snapshot_linear(self) -> np.ndarray:
        with self._lock:
            return self._linear.copy()

    def snapshot_db(self) -> np.ndarray:
        with self._lock:
            return self._db.copy()


class FileSource:
    def __init__(self, filename: Path, target_rate: int):
        rate, audio = wav.read(str(filename))
        if rate != target_rate:
            raise ValueError(
                f"{filename} has sample rate {rate} Hz, but output rate is {target_rate} Hz. "
                "Use --rate to match the file, or resample it first."
            )

        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if np.issubdtype(audio.dtype, np.integer):
            max_value = float(np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float64) / max_value
        else:
            audio = audio.astype(np.float64)

        self.audio = np.clip(audio, -1.0, 1.0)
        self.position = 0

    def read(self, frames: int) -> np.ndarray:
        block = np.zeros(frames, dtype=np.float64)
        remaining = len(self.audio) - self.position
        if remaining <= 0:
            return block
        take = min(frames, remaining)
        block[:take] = self.audio[self.position : self.position + take]
        self.position += take
        return block


class EqualizerAudio:
    def __init__(self, args: argparse.Namespace, gains: SharedGains, status: queue.Queue[str]):
        self.args = args
        self.gains = gains
        self.status = status
        self.filter_bank = MDCTFilterBank(args.bands)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.file_source = None

        if args.input != "mic":
            self.file_source = FileSource(Path(args.input), args.rate)

    def start(self) -> None:
        kwargs = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": self.args.rate,
            "output": True,
            "frames_per_buffer": self.args.bands,
            "stream_callback": self._callback,
        }
        if self.file_source is None:
            kwargs["input"] = True
            if self.args.input_device is not None:
                kwargs["input_device_index"] = self.args.input_device
            self.status.put("input: microphone")
        else:
            kwargs["input"] = False
            self.status.put(f"input: {self.args.input}")

        if self.args.output_device is not None:
            kwargs["output_device_index"] = self.args.output_device

        self.stream = self.audio.open(**kwargs)
        self.stream.start_stream()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.audio.terminate()

    def _read_input(self, in_data: bytes | None, frame_count: int) -> np.ndarray:
        if self.file_source is not None:
            return self.file_source.read(frame_count)
        if in_data is None:
            return np.zeros(frame_count, dtype=np.float64)
        shorts = struct.unpack("h" * frame_count, in_data)
        return np.asarray(shorts, dtype=np.float64) / 32768.0

    def _callback(self, in_data, frame_count, time_info, status_flags):
        samples = self._read_input(in_data, frame_count)
        gains = self.gains.snapshot_linear()

        out = np.zeros_like(samples)
        for start in range(0, frame_count, self.args.bands):
            block = samples[start : start + self.args.bands]
            if len(block) < self.args.bands:
                padded = np.zeros(self.args.bands, dtype=np.float64)
                padded[: len(block)] = block
                block = padded
            out[start : start + self.args.bands] = self.filter_bank.process_block(block, gains)[
                : len(out[start : start + self.args.bands])
            ]

        out = np.clip(out * self.args.output_gain, -1.0, 1.0)
        data = (out * 32767.0).astype(np.int16).tobytes()
        return data, pyaudio.paContinue


class EqualizerGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.gains = SharedGains(args.bands)
        self.status = queue.Queue()
        self.audio = EqualizerAudio(args, self.gains, self.status)
        self.sliders: list[tk.Scale] = []
        self.value_labels: list[ttk.Label] = []

        root.title(f"MDCT Equalizer ({args.bands} subbands)")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.audio.start()
        self._poll_status()

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(header, text="starting audio ...")
        self.status_label.grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Reset", command=self.reset).grid(row=0, column=1, sticky="e")

        sliders = ttk.Frame(main)
        sliders.grid(row=1, column=0, pady=(12, 0), sticky="nsew")

        nyquist = self.args.rate / 2.0
        for band in range(self.args.bands):
            frame = ttk.Frame(sliders, padding=(3, 0))
            frame.grid(row=0, column=band, sticky="ns")
            low = band * nyquist / self.args.bands
            high = (band + 1) * nyquist / self.args.bands
            ttk.Label(frame, text=self._band_label(low, high), width=8, anchor="center").grid(
                row=0, column=0
            )
            slider = tk.Scale(
                frame,
                from_=18,
                to=-60,
                resolution=0.5,
                orient=tk.VERTICAL,
                length=280,
                showvalue=False,
                command=lambda value, b=band: self._set_gain(b, value),
            )
            slider.set(0)
            slider.grid(row=1, column=0)
            value_label = ttk.Label(frame, text="0.0 dB", width=8, anchor="center")
            value_label.grid(row=2, column=0)
            self.sliders.append(slider)
            self.value_labels.append(value_label)

    def _set_gain(self, band: int, value: str) -> None:
        value_db = float(value)
        self.gains.set_db(band, value_db)
        self.value_labels[band].configure(text=f"{value_db:+.1f} dB")

    def reset(self) -> None:
        for slider in self.sliders:
            slider.set(0)

    def _poll_status(self) -> None:
        try:
            while True:
                self.status_label.configure(text=self.status.get_nowait())
        except queue.Empty:
            pass
        self.root.after(200, self._poll_status)

    def close(self) -> None:
        self.audio.stop()
        self.root.destroy()

    @staticmethod
    def _band_label(low: float, high: float) -> str:
        if high < 1000:
            return f"{int(low)}-{int(high)}"
        return f"{low / 1000:.1f}-{high / 1000:.1f}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time MDCT equalizer")
    parser.add_argument("--bands", type=int, choices=(8, 16), default=16)
    parser.add_argument("--input", default="mic", help="'mic' or a mono/stereo WAV file")
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="sample rate in Hz; defaults to 32000 for mic, or the WAV file rate",
    )
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument(
        "--output-gain",
        type=float,
        default=0.7,
        help="linear gain after synthesis to avoid clipping",
    )
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def list_devices() -> None:
    audio = pyaudio.PyAudio()
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            print(
                f"{index}: {info['name']} "
                f"in={info['maxInputChannels']} out={info['maxOutputChannels']} "
                f"default_rate={info['defaultSampleRate']}"
            )
    finally:
        audio.terminate()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0
    if args.rate is None:
        if args.input == "mic":
            args.rate = 32000
        else:
            args.rate = int(wav.read(str(Path(args.input)), mmap=True)[0])

    root = tk.Tk()
    try:
        EqualizerGUI(root, args)
        root.mainloop()
    except Exception as exc:
        root.destroy()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
