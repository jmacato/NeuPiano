"""Fit what the strings and soundboard modes miss in the first 120 ms as fast-decaying modes: below 1.5 kHz anywhere (hammer knock, string build-up),
above it at each partial's frequency (the prompt part of a two-stage decay)."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from modal_piano import RATE, raw_recordings

LAYERS, ATTACK = 10, 48
CEILING, TOP, FACTOR, WINDOW, LEAD, SPACING = 1500., 1950., 8, .12, .02, 4.
TAUS = (.0015, .003, .006, .012, .024)
LIMIT, GAIN = 96, .001
PROMPT, STRIDE, LOWEST, ATOMS, SHARE = (.003, .006, .012, .024, .048), 48, 150., 2, .2


def dictionary():
    t = np.arange(int(WINDOW * RATE) // FACTOR) * FACTOR / RATE
    ramp = np.where(t * RATE < ATTACK, .5 - .5 * np.cos(np.pi * t * RATE / ATTACK), 1)
    frequency, tau = (grid.ravel() for grid in np.meshgrid(np.arange(20, TOP + 1, SPACING), TAUS, indexing='ij'))
    envelope = ramp[:, None] * np.exp(-t[:, None] / tau[None])
    phase = 2 * np.pi * frequency[None] * t[:, None]
    return frequency, tau, envelope * np.cos(phase), envelope * np.sin(phase)


def tonal(model, k, layer, channel, frames):
    t = np.arange(frames) / RATE
    start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
    rows = slice(start, start + count)
    f, beat, theta = model['frequency'][rows, None], model['beat'][rows, None], model['theta'][rows, None]
    audio = (model['amplitude1'][rows, None] * np.exp(-model['decay1'][rows, None] * t) * np.cos(2 * np.pi * (f + beat / 2) * t + theta)).sum(0)
    audio += (model['amplitude2'][rows, None] * np.exp(-model['decay2'][rows, None] * t) * np.cos(2 * np.pi * (f - beat / 2) * t + theta - model['phase'][rows, None])).sum(0)
    first, count = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
    rows = slice(first, first + count)
    audio += (model['mode_amplitude'][rows, None] * np.exp(-model['mode_decay'][rows, None] * t) *
              np.cos(2 * np.pi * model['mode_frequency'][rows, None] * t + model['mode_phase'][rows, None])).sum(0)
    audio[:ATTACK] *= .5 - .5 * np.cos(np.pi * np.arange(ATTACK) / ATTACK)
    return audio


def residual(x, model, k, layer, channel):
    """The recording minus strings and modes from the onset with LEAD of real lead-in and after, and its part below the ceiling at RATE / FACTOR."""
    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
    lead, span = int(LEAD * RATE), int(WINDOW * RATE)
    segment = np.concatenate([np.zeros(lead), x])[onset:onset + 2 * lead + span]
    segment[lead:] -= tonal(model, k, layer, channel, len(segment) - lead)
    low = sosfiltfilt(butter(8, CEILING, 'lowpass', fs=RATE, output='sos'), segment)
    return segment, low[lead:lead + span:FACTOR]


def modes_audio(rows, frames):
    t = np.arange(frames) / RATE
    audio = (rows[:, 1, None] * np.exp(-rows[:, 2, None] * t) * np.cos(2 * np.pi * rows[:, 0, None] * t + rows[:, 3, None])).sum(0)
    audio[:ATTACK] *= .5 - .5 * np.cos(np.pi * np.arange(ATTACK) / ATTACK)
    return audio


def prompt(segment, frequencies, f0):
    """Up to ATOMS damped modes per partial above the ceiling, chosen greedily from PROMPT decay times on the residual demodulated at the partial."""
    lead, span = int(LEAD * RATE), int(WINDOW * RATE)
    width = min(150., .4 * f0)
    t = (np.arange(len(segment)) - lead) / RATE
    grid = np.arange(span // STRIDE) * STRIDE / RATE
    ramp = np.where(grid * RATE < ATTACK, .5 - .5 * np.cos(np.pi * grid * RATE / ATTACK), 1)
    atoms = ramp[:, None] * np.exp(-grid[:, None] / np.array(PROMPT)[None])
    sos = butter(4, width, 'lowpass', fs=RATE, output='sos')
    rows = []
    for f in frequencies[(frequencies > CEILING) & (frequencies < .45 * RATE)]:
        shifted = segment * np.exp(-2j * np.pi * f * t)
        base = (sosfiltfilt(sos, shifted.real) + 1j * sosfiltfilt(sos, shifted.imag))[lead:lead + span:STRIDE]
        chosen, rest, total = [], base, float(np.vdot(base, base).real)
        for _ in range(ATOMS):
            gain = np.abs(atoms.T @ rest) ** 2 / (atoms ** 2).sum(0)
            gain[chosen] = 0
            best = int(np.argmax(gain))
            if gain[best] < SHARE * total:
                break
            chosen.append(best)
            coefficients = np.linalg.lstsq(atoms[:, chosen], base, rcond=None)[0]
            rest = base - atoms[:, chosen] @ coefficients
        rows += [(f, 2 * abs(c), 1 / PROMPT[i], np.angle(c)) for i, c in zip(chosen, coefficients if chosen else [])]
    return np.array(rows).reshape(-1, 4)


def pursue(r, frequency, tau, c, s):
    """Orthogonal matching pursuit over damped cos/sin pairs; returns rows of (frequency, amplitude, decay, phase)."""
    cc, ss, cs = (c * c).sum(0), (s * s).sum(0), (c * s).sum(0)
    det = np.maximum(cc * ss - cs ** 2, 1e-9 * cc * ss + 1e-30)
    total = float(r @ r)
    chosen, rest, coefficients = [], r.copy(), np.zeros(0)
    while len(chosen) < LIMIT and total > 0:
        a, b = c.T @ rest, s.T @ rest
        gain = (ss * a * a - 2 * cs * a * b + cc * b * b) / det
        gain[chosen] = 0
        best = int(np.argmax(gain))
        if gain[best] < GAIN * total:
            break
        chosen.append(best)
        basis = np.concatenate([c[:, chosen], s[:, chosen]], 1)
        gram = basis.T @ basis
        coefficients = np.linalg.solve(gram + 1e-9 * np.trace(gram) / len(gram) * np.eye(len(gram)), basis.T @ r)
        rest = r - basis @ coefficients
    n = len(chosen)
    a, b = coefficients[:n], coefficients[n:]
    return np.stack([frequency[chosen], np.hypot(a, b), 1 / tau[chosen], np.arctan2(-b, a)], 1), 1 - float(rest @ rest) / max(total, 1e-30)


def fit_key(job):
    path, cache, k, note = job
    model = np.load(path)
    layers = raw_recordings(cache, note)
    atoms = dictionary()
    f0 = float(model['f0'][k])
    result = []
    for layer in range(LAYERS):
        for channel in range(2):
            segment, low = residual(layers[layer][:, channel], model, k, layer, channel)
            rows, captured = pursue(low, *atoms)
            if f0 >= LOWEST:
                lead = int(LEAD * RATE)
                segment[lead:] -= modes_audio(rows, len(segment) - lead)
                start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
                rows = np.concatenate([rows, prompt(segment, model['frequency'][start:start + count].astype(np.float64), f0)])
            result.append((rows, captured))
    return k, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--processes', type=int, default=6)
    args = parser.parse_args()
    model = dict(np.load(args.model))
    notes = [int(n) for n in model['notes']]
    started = time.monotonic()
    fitted = {}
    with Pool(args.processes) as pool:
        for k, result in pool.imap_unordered(fit_key, [(args.model, args.cache, k, n) for k, n in enumerate(notes)]):
            fitted[k] = result
            print(f'note {notes[k]}: {sum(len(rows) for rows, _ in result)} attack modes, {np.mean([c for _, c in result]):.0%} of the residual', flush=True)
    names = ('frequency', 'amplitude', 'decay', 'phase')
    rows, counts = [], np.zeros_like(model['mode_counts'])
    for k in range(len(notes)):
        for index, (attack, _) in enumerate(fitted[k]):
            layer, channel = divmod(index, 2)
            first, count = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
            rows.append(np.concatenate([np.stack([model[f'mode_{name}'][first:first + count] for name in names], 1), attack]))
            counts[k, layer, channel] = len(rows[-1])
    rows = np.concatenate(rows).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(counts.ravel())[:-1]]).reshape(counts.shape).astype(np.int32)
    model.update(mode_offsets=offsets, mode_counts=counts, **{f'mode_{name}': rows[:, i] for i, name in enumerate(names)})
    temporary = args.output.with_suffix('.partial.npz')
    np.savez(temporary, **model)
    temporary.replace(args.output)
    print(f'wrote {args.output}: {len(rows)} modes in {time.monotonic() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
