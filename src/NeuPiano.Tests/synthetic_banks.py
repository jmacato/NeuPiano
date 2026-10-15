"""Write a synthetic modal bank, control net and release bank for middle C from stiff-string formulas and random weights, so the engine tests need no recordings.

Usage: synthetic_banks.py OUTPUT_DIR, then make_modal_reference.py OUTPUT_DIR/modal_bank.npz OUTPUT_DIR/control_net.npz OUTPUT_DIR/release_bank.npz"""
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'training'))
from fit_modal_gpu import band_edges, noise_filters
from modal_piano import partial_frequencies

NOTE, F0, INHARMONICITY = 60, 261.6, 4e-4
THRESHOLDS = [12, 24, 36, 48, 60, 72, 85, 105, 118, 127]
RELEASE_THRESHOLDS = [400, 1500, 5000, 32767, 60, 110, 127]
NAMES = ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase', 'theta')
MODES = ('frequency', 'amplitude', 'decay', 'phase')


def bank(rng, thresholds, seconds, partials, scale, decay, delays=None):
    layers, bands = len(thresholds), len(band_edges())
    shape = (1, layers, 2)
    offsets, counts, mode_offsets, mode_counts = (np.zeros(shape, np.int32) for _ in range(4))
    noise = np.zeros(shape + (bands, 6), np.float32)
    rows, modes = [], []
    frequencies = partial_frequencies(F0, INHARMONICITY, partials)
    n = np.arange(1, partials + 1)
    for layer in range(layers):
        if seconds[layer] == 0:
            offsets[0, layer], mode_offsets[0, layer] = sum(map(len, rows)), sum(map(len, modes))
            continue
        level = scale * 10 ** (-(layers - 1 - layer) * 2 / 20)
        for channel in range(2):
            a1 = level * .05 / n * (1 + layer / (layers - 1) * n / 10) * rng.uniform(.7, 1.3, partials)
            d1 = decay * (.4 + .08 * n) * rng.uniform(.8, 1.2, partials)
            strings = np.stack([frequencies, a1, a1 * rng.uniform(.4, .9, partials), d1, d1 * rng.uniform(.3, .7, partials),
                                rng.uniform(0, 1.2, partials), rng.uniform(-np.pi, np.pi, partials), rng.uniform(0, 2 * np.pi, partials)], 1)
            body = np.stack([rng.uniform(60, .75 * F0, 30), level * rng.uniform(.001, .006, 30), rng.uniform(3, 15, 30), rng.uniform(-np.pi, np.pi, 30)], 1)
            offsets[0, layer, channel], counts[0, layer, channel] = sum(map(len, rows)), partials
            mode_offsets[0, layer, channel], mode_counts[0, layer, channel] = sum(map(len, modes)), len(body)
            rows.append(strings)
            modes.append(body)
            noise[0, layer, channel] = np.stack([level ** 2 * 1e-3 * rng.uniform(.1, 1, bands), rng.uniform(.002, .01, bands),
                                                 level ** 2 * 1e-5 * rng.uniform(.1, 1, bands), rng.uniform(.05, .3, bands),
                                                 np.full(bands, 1e-9), np.full(bands, .002)], 1)
    rows, modes = np.concatenate(rows).astype(np.float32), np.concatenate(modes).astype(np.float32)
    arrays = {name: rows[:, i] for i, name in enumerate(NAMES)} | {f'mode_{name}': modes[:, i] for i, name in enumerate(MODES)}
    arrays.update(notes=np.array([NOTE], np.int32), offsets=offsets, counts=counts, mode_offsets=mode_offsets, mode_counts=mode_counts,
                  noise=noise, seconds=np.array([seconds], np.float32), levels=np.ones((1, layers), np.float32),
                  thresholds=np.array(thresholds, np.int32), sos=noise_filters(),
                  f0=np.array([F0], np.float32), inharmonicity=np.array([INHARMONICITY], np.float32))
    if delays is not None:
        arrays['delays'] = np.full((1, layers), delays, np.float32)
    return arrays


def control_net(rng, harmonics=32, dimensions=16, hidden=64, keys=88, layers=10):
    bands = len(band_edges())
    sizes = [dimensions + 3 + 2 * harmonics, hidden, hidden, 2 * 2 * bands]
    net = {'embedding': rng.normal(0, .5, (keys, layers, dimensions)).astype(np.float32),
           'lows': np.array([low for low, _ in band_edges()], np.float32),
           'scalars': np.array([harmonics, .05, 14, 5, len(sizes) - 1, 21, keys, .02], np.float32)}
    for i, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
        gain = .3 if i == len(sizes) - 2 else 1
        net[f'weight{i}'] = (rng.normal(0, gain / np.sqrt(inputs), (outputs, inputs))).astype(np.float32)
        net[f'bias{i}'] = rng.normal(0, .05, outputs).astype(np.float32)
    return net


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    np.savez(output / 'modal_bank.npz', **bank(rng, THRESHOLDS, [8.0] * len(THRESHOLDS), 40, 1, 1))
    np.savez(output / 'control_net.npz', **control_net(rng))
    np.savez(output / 'release_bank.npz', **bank(rng, RELEASE_THRESHOLDS, [0, 2.5, 2.5, 1, 2.5, 2.5, 2], 30, .2, 2, delays=.0005))
    print('wrote synthetic banks to', output)


if __name__ == '__main__':
    main()
