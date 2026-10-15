"""Refit every string partial in every raw recording channel from its own narrowband envelope: centre frequency, two-string envelope and starting phase."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from fit_modal_gpu import Throttle, fit_envelopes
from fit_phases import phases
from modal_piano import HOP, RATE, raw_recordings

LAYERS, ORDER, SPAN, WIDEST, LIMIT, FRAMES = 10, 3, 1., 60., 30., 701


def esprit(z, order):
    rows = min(len(z) // 2, max(3 * order, 40))
    hankel = np.lib.stride_tricks.sliding_window_view(z, len(z) - rows + 1)[:rows]
    basis = np.linalg.svd(hankel, full_matrices=False)[0][:, :order]
    return np.linalg.eigvals(np.linalg.lstsq(basis[:-1], basis[1:], rcond=None)[0])


def narrowband(spectrum, duration, n, f, width):
    """Complex envelope of the band f +- width, its sample rate, and the frequency it is centred on."""
    lo, hi = max(int(np.ceil((f - width) * duration)), 1), min(int(np.floor((f + width) * duration)), len(spectrum) - 1)
    if hi - lo < 8:
        return None, None, None
    bins = np.arange(lo, hi + 1)
    taper = np.ones(len(bins))
    edge = max(1, len(bins) // 6)
    ramp = .5 - .5 * np.cos(np.pi * np.arange(edge) / edge)
    taper[:edge], taper[-edge:] = ramp, ramp[::-1]
    size = 1 << int(np.ceil(np.log2(4 * len(bins))))
    center = int(round(f * duration))
    baseband = np.zeros(size, complex)
    baseband[(bins - center) % size] = spectrum[bins] * taper
    return np.fft.ifft(baseband) * size * 2 / n, size / duration, center / duration


def centre(z, rate, fc, width, skip):
    t = np.arange(len(z)) / rate
    window = (t >= skip) & (t <= skip + SPAN)
    if window.sum() < 3 * ORDER + 4:
        return None
    p = esprit(z[window], ORDER)
    decay, offset = -np.log(np.abs(p)) * rate, np.angle(p) * rate / (2 * np.pi)
    keep = (decay > -1) & (np.abs(offset) < width)
    if not keep.any():
        return None
    basis = np.exp((-decay[keep] + 2j * np.pi * offset[keep])[None] * t[window][:, None])
    amplitude = np.linalg.lstsq(basis, z[window], rcond=None)[0]
    energy = np.abs(amplitude) ** 2 * np.exp(-2 * decay[keep] * skip)
    return fc + np.sum(energy * offset[keep]) / energy.sum() if energy.sum() > 0 else None


def measure_channel(x, frequencies, f0):
    """Corrected frequencies, 50 Hz envelopes |z|^2 and noise floors for one channel's partials."""
    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
    segment = x[onset:]
    spectrum = np.fft.rfft(segment)
    duration, n = len(segment) / RATE, len(segment)
    width = min(.45 * f0, WIDEST)
    skip = max(.01, 1.5 / width)
    frames = np.arange(FRAMES) * HOP / RATE
    corrected = frequencies.copy()
    envelopes = np.zeros((len(frequencies), FRAMES))
    floors = np.full(len(frequencies), 1e-30)
    for i, f in enumerate(frequencies):
        z, rate, fc = narrowband(spectrum, duration, n, f, width)
        if z is None:
            continue
        g = centre(z, rate, fc, width, skip)
        if g is not None and abs(1200 * np.log2(g / f)) < LIMIT:
            corrected[i] = g
            z, rate, fc = narrowband(spectrum, duration, n, g, width)
        t = np.arange(len(z)) / rate
        power = np.convolve(np.abs(z) ** 2, np.ones(5) / 5, 'same')
        envelopes[i] = np.interp(frames, t, power, right=0)
        tail = (t > duration - 2) & (t < duration - .3)
        floors[i] = max(np.median(power[tail]) if tail.any() else 0, power.max() * 1e-7, 1e-30)
    return corrected, envelopes, floors, skip


def measure_key(job):
    path, cache, k, note = job
    model = np.load(path)
    layers = raw_recordings(cache, note)
    result = []
    for layer in range(LAYERS):
        for channel in range(2):
            start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
            result.append((start, *measure_channel(layers[layer][:, channel], model['frequency'][start:start + count].astype(np.float64),
                                                   float(model['f0'][k]))))
    return k, result


def phase_key(job):
    path, cache, k, note, pairs = job
    model = np.load(path)
    layers = raw_recordings(cache, note)
    result = []
    for index, (start, rows) in enumerate(pairs):
        layer, channel = divmod(index, 2)
        first, count = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
        modes = np.stack([model[f'mode_{name}'][first:first + count].astype(np.float64) for name in ('frequency', 'amplitude', 'decay', 'phase')], 1)
        result.append((start, phases(layers[layer][:, channel], rows, modes)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--processes', type=int, default=6)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--duty', type=float, default=.5)
    parser.add_argument('--memory-gib', type=float, default=2.)
    args = parser.parse_args()
    if not 0 < args.duty <= .5:
        raise ValueError('GPU duty must stay at or below 50%')
    model = dict(np.load(args.model))
    notes = [int(n) for n in model['notes']]
    device = torch.device('mps')
    throttle = Throttle(args.duty, args.memory_gib)
    table = torch.ones(2401, device=device)
    t = torch.arange(FRAMES, device=device, dtype=torch.float32) * HOP / RATE
    fitted = {name: model[name].copy() for name in ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase')}
    started = time.monotonic()
    pairs_by_key = {}
    with Pool(args.processes) as pool:
        for k, result in pool.imap_unordered(measure_key, [(args.model, args.cache, k, n) for k, n in enumerate(notes)]):
            count = max(len(r[1]) for r in result)
            energy = np.zeros((len(result), count, FRAMES), np.float32)
            floor = np.ones((len(result), count, FRAMES), np.float32)
            weight = np.zeros((len(result), 1, FRAMES), np.float32)
            for c, (start, frequency, envelopes, floors, skip) in enumerate(result):
                energy[c, :len(frequency)] = envelopes
                floor[c, :len(frequency)] = floors[:, None]
                length = int(round(model['seconds'][k, c // 2] * 50))
                weight[c, 0, int(np.ceil(skip * 50)):length] = 1
                weight[c, 0] *= np.minimum(1, 1 / np.maximum(np.arange(FRAMES) * HOP / RATE, 1e-3))
            budget = max(1, int(6e6 // (5 * count * FRAMES)))
            chunks = [fit_envelopes(torch.tensor(energy[c:c + budget], device=device), torch.tensor(floor[c:c + budget], device=device),
                                    torch.tensor(weight[c:c + budget], device=device), t, table, throttle, args.steps)
                      for c in range(0, len(result), budget)]
            parameters = np.concatenate([chunk[0].cpu().numpy() for chunk in chunks])
            exists = np.concatenate([chunk[1].cpu().numpy() for chunk in chunks])
            pairs = []
            for c, (start, frequency, envelopes, floors, skip) in enumerate(result):
                rows = slice(start, start + len(frequency))
                p, ok = parameters[c, :len(frequency)], exists[c, :len(frequency)]
                fitted['frequency'][rows] = frequency
                for column, name in ((0, 'amplitude1'), (1, 'amplitude2')):
                    fitted[name][rows] = np.where(ok, np.exp(p[:, column]), fitted[name][rows])
                for column, name in ((2, 'decay1'), (3, 'decay2'), (4, 'beat'), (5, 'phase')):
                    fitted[name][rows] = np.where(ok, p[:, column], fitted[name][rows])
                pairs.append((start, np.stack([fitted[name][rows].astype(np.float64) for name in
                                               ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase')], 1)))
            pairs_by_key[k] = pairs
            print(f'note {notes[k]}: refit {sum(len(r[1]) for r in result)} partials; {throttle.report()}', flush=True)
        theta = model['theta'].copy()
        for result in pool.imap_unordered(phase_key, [(args.model, args.cache, k, n, pairs_by_key[k]) for k, n in enumerate(notes)]):
            for start, values in result:
                theta[start:start + len(values)] = values
    model.update({name: values.astype(np.float32) for name, values in fitted.items()}, theta=theta.astype(np.float32))
    temporary = args.output.with_suffix('.partial.npz')
    np.savez(temporary, **model)
    temporary.replace(args.output)
    print(f'wrote {args.output} in {time.monotonic() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
