"""Measure each string partial's starting phase in every raw recording channel and add it to the modal export."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from modal_piano import RATE, raw_recordings

LAYERS = 10
SKIP, WINDOW = .002, .15


def phases(x, rows, modes):
    """Starting phase of each partial, from a least-squares fit of the fitted string pairs to the recording after the onset."""
    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
    t = np.arange(int(WINDOW * RATE)) / RATE
    segment = x[onset:onset + len(t)].copy()
    t = t[:len(segment)]
    for f, a, d, p in modes:
        segment -= a * np.exp(-d * t) * np.cos(2 * np.pi * f * t + p)
    f, a1, a2, d1, d2, beat, phi = rows.T
    template = (a1[None] * np.exp(-d1[None] * t[:, None]) * np.exp(2j * np.pi * (f + beat / 2)[None] * t[:, None])
                + a2[None] * np.exp(-d2[None] * t[:, None]) * np.exp(2j * np.pi * (f - beat / 2)[None] * t[:, None] - 1j * phi[None]))
    used = t >= SKIP
    design = np.concatenate([template.real, -template.imag], 1)[used]
    coefficients = np.linalg.lstsq(design, segment[used], rcond=None)[0]
    return np.angle(coefficients[:len(f)] + 1j * coefficients[len(f):])


def fit_key(job):
    path, cache, k, note = job
    model = np.load(path)
    layers = raw_recordings(cache, note)
    result = []
    for layer in range(LAYERS):
        for channel in range(2):
            start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
            rows = np.stack([model[name][start:start + count].astype(np.float64) for name in
                             ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase')], 1)
            first, modes = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
            mode_rows = np.stack([model[f'mode_{name}'][first:first + modes].astype(np.float64) for name in
                                  ('frequency', 'amplitude', 'decay', 'phase')], 1)
            result.append((start, phases(layers[layer][:, channel], rows, mode_rows)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--processes', type=int, default=6)
    args = parser.parse_args()
    model = dict(np.load(args.model))
    theta = np.zeros(len(model['frequency']), np.float32)
    started = time.monotonic()
    with Pool(args.processes) as pool:
        for result in pool.imap_unordered(fit_key, [(args.model, args.cache, k, int(n)) for k, n in enumerate(model['notes'])]):
            for start, values in result:
                theta[start:start + len(values)] = values
    print(f'measured {len(theta)} partial phases in {time.monotonic() - started:.0f} s', flush=True)
    model['theta'] = theta
    temporary = args.model.with_suffix('.partial.npz')
    np.savez(temporary, **model)
    temporary.replace(args.model)
    print(f'wrote {args.model}', flush=True)


if __name__ == '__main__':
    main()
