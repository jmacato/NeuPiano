"""Fit the hammer and soundboard knock below each recording's fundamental as damped modes and add them, with the per-layer playback levels from cache/levels.npy, to the modal export."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from fit_modal_gpu import synthesize
from modal_piano import RATE, raw_recordings

LAYERS = 10
WINDOW, ORDER, LOWEST, CEILING = 1.0, 160, 40., 1500.


def poles(x, order):
    rows = min(len(x) // 2, 3 * order)
    hankel = np.lib.stride_tricks.sliding_window_view(x, len(x) - rows + 1)[:rows]
    basis = np.linalg.svd(hankel, full_matrices=False)[0][:, :order]
    return np.linalg.eigvals(np.linalg.lstsq(basis[:-1], basis[1:], rcond=None)[0])


def fit_modes(x, f0):
    """Damped modes (frequency, amplitude, decay, phase) for the part of one channel below 0.75 f0, timed from the hammer onset."""
    cutoff = min(.75 * f0, CEILING)
    if cutoff < LOWEST:
        return np.zeros((0, 4))
    low = sosfiltfilt(butter(8, cutoff, 'lowpass', fs=RATE, output='sos'), x)
    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
    factor = max(1, int(RATE // (4 * cutoff)))
    segment = resample_poly(low[onset:onset + int(WINDOW * RATE)], 1, factor)
    rate = RATE / factor
    z = poles(segment, min(ORDER, len(segment) // 4))
    frequency, decay = np.angle(z) * rate / (2 * np.pi), -np.log(np.abs(z) + 1e-300) * rate
    z = z[(frequency > 0) & (frequency < cutoff) & (decay >= 1)]
    t = np.arange(len(segment))
    basis = z[None] ** t[:, None]
    coefficients = np.linalg.lstsq(np.concatenate([basis.real, -basis.imag], 1), segment, rcond=None)[0]
    complex_amplitude = coefficients[:len(z)] + 1j * coefficients[len(z):]
    frequency, decay = np.angle(z) * rate / (2 * np.pi), -np.log(np.abs(z)) * rate
    return np.stack([frequency, np.abs(complex_amplitude), decay, np.angle(complex_amplitude)], 1)


def fit_key(job):
    cache, note, f0 = job
    layers = raw_recordings(cache, note)
    return note, [fit_modes(layers[layer][:, channel], f0) for layer in range(LAYERS) for channel in range(2)]


def mode_audio(rows, frames, device):
    frequency, amplitude, decay, phase = rows.T
    with np.errstate(divide='ignore'):
        partials = np.stack([np.log(amplitude), np.full_like(amplitude, -np.inf), decay, decay, np.zeros_like(decay), np.zeros_like(decay)], 1)
    return synthesize(torch.tensor(partials, dtype=torch.float32, device=device), torch.ones(len(rows), device=device),
                      frequency.astype(np.float64), frames, device, 0, theta=phase.astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--processes', type=int, default=6)
    args = parser.parse_args()
    model = dict(np.load(args.model))
    notes = list(model['notes'])
    started = time.monotonic()
    with Pool(args.processes) as pool:
        fitted = dict(pool.map(fit_key, [(args.cache, note, float(model['f0'][k])) for k, note in enumerate(notes)]))
    counts = np.array([[len(m) for m in fitted[n]] for n in notes]).reshape(len(notes), LAYERS, 2)
    print(f'fitted {counts.sum()} modes in {time.monotonic() - started:.0f} s, up to {counts.max()} per channel', flush=True)
    offsets = np.concatenate([[0], np.cumsum(counts.ravel())[:-1]]).reshape(counts.shape).astype(np.int32)
    rows = np.concatenate([m for n in notes for m in fitted[n]]).astype(np.float32)
    levels_path = args.cache / 'levels.npy'
    levels = np.load(levels_path) if levels_path.exists() else np.ones((len(notes), LAYERS))
    model.update(mode_offsets=offsets, mode_counts=counts.astype(np.int32), mode_frequency=rows[:, 0],
                 mode_amplitude=rows[:, 1], mode_decay=rows[:, 2], mode_phase=rows[:, 3], levels=levels.astype(np.float32))
    temporary = args.output.with_suffix('.partial.npz')
    np.savez(temporary, **model)
    temporary.replace(args.output)
    print(f'wrote {args.output}: {len(rows)} modes', flush=True)


if __name__ == '__main__':
    main()
