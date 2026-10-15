"""Write reference_modal.npz and reference_release.npz: middle C from a modal bank, its control net and release bank, plus Python renders the C# engine is checked against.

Usage: make_modal_reference.py [modal_bank.npz control_net.npz release_bank.npz]; synthetic_banks.py writes a bank set made from scratch."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'training'))
from fit_modal_gpu import render
from modal_piano import HOP, RATE

source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'banks/modal_bank.npz'
control = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'banks/control_net.npz'
model = dict(np.load(source))
net = dict(np.load(control))


def controls(note, layer, frames):
    harmonics, origin, span, bound, layers, first, keys = net['scalars'][:7].astype(np.float64)
    start = float(net['scalars'][7]) if len(net['scalars']) > 7 else 0.
    key = note - int(first)
    t = np.clip(np.arange(frames) * HOP / RATE, 0, span)
    u = (np.log1p(t / origin) / np.log1p(span / origin))[:, None]
    k = np.arange(1, int(harmonics) + 1) * np.pi
    static = np.concatenate([net['embedding'][key, layer], [key / (keys - 1), layer / 9]])
    x = np.concatenate([np.broadcast_to(static, (frames, len(static))), u, np.sin(u * k), np.cos(u * k)], 1)
    for i in range(int(layers)):
        x = x @ net[f'weight{i}'].T.astype(np.float64) + net[f'bias{i}']
        x = x / (1 + np.exp(-x)) if i < layers - 1 else bound * np.tanh(x / bound)
    return x * (np.clip(t / start, 0, 1)[:, None] if start > 0 else 1)


def expected_noise_rms(note, layer, channel, gains=None):
    k = int(np.flatnonzero(model['notes'] == note)[0])
    t = np.arange(96000) / RATE
    impulse = np.zeros(RATE)
    impulse[0] = 1
    power = np.zeros(len(t))
    for band, (knock, fast, ring, slow, rest, *rise) in enumerate(model['noise'][k, layer, channel]):
        level = (knock * np.exp(-t / fast) + ring * np.exp(-t / slow) + rest) * np.sum(sosfilt(model['sos'][band], impulse) ** 2)
        if rise:
            level *= -np.expm1(-t / rise[0])
        if gains is not None:
            level *= np.exp(2 * np.interp(t, np.arange(gains.shape[-1]) * HOP / RATE, gains[1, band]))
        power += level
    return np.sqrt(power.reshape(20, 4800).mean(1)).astype(np.float32)


k = int(np.flatnonzero(model['notes'] == 60)[0])
first, last = model['offsets'][k].min(), (model['offsets'][k] + model['counts'][k]).max()
subset = {name: model[name][first:last] for name in ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase', 'theta')}
subset.update(notes=np.array([60], np.int32), offsets=model['offsets'][k:k + 1] - first, counts=model['counts'][k:k + 1],
              noise=model['noise'][k:k + 1], seconds=model['seconds'][k:k + 1], thresholds=model['thresholds'], sos=model['sos'],
              levels=model['levels'][k:k + 1])
first, last = model['mode_offsets'][k].min(), (model['mode_offsets'][k] + model['mode_counts'][k]).max()
subset.update({f'mode_{name}': model[f'mode_{name}'][first:last] for name in ('frequency', 'amplitude', 'decay', 'phase')},
              mode_offsets=model['mode_offsets'][k:k + 1] - first, mode_counts=model['mode_counts'][k:k + 1])
level = float(model['levels'][k, 9])
for channel, name in enumerate(('left', 'right')):
    subset[f'tonal_{name}'] = (level * render(subset, 60, 9, channel, noise=False)[:48000]).astype(np.float32)
subset['noise_rms'] = level * expected_noise_rms(60, 9, 0)

frames = 701
values = controls(60, 9, frames)
curves = values.reshape(frames, 2, 2, -1).transpose(1, 2, 3, 0)
for channel, name in enumerate(('left', 'right')):
    subset[f'controlled_{name}'] = (level * render(subset, 60, 9, channel, noise=False, gains=curves[channel])[:48000]).astype(np.float32)
subset['controlled_noise_rms'] = level * expected_noise_rms(60, 9, 0, curves[0])
subset['controls'] = values.astype(np.float32)
subset.update({name: value for name, value in net.items()})
np.savez(Path(__file__).resolve().parent / 'reference_modal.npz', **subset)
print('wrote reference_modal.npz from', source, 'and', control)

releases = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'banks/release_bank.npz'
if releases.exists():
    bank = dict(np.load(releases))
    k = int(np.flatnonzero(bank['notes'] == 60)[0])
    first, last = bank['offsets'][k].min(), (bank['offsets'][k] + bank['counts'][k]).max()
    part = {name: bank[name][first:last] for name in ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase', 'theta')}
    part.update(notes=np.array([60], np.int32), offsets=bank['offsets'][k:k + 1] - first, counts=bank['counts'][k:k + 1], noise=bank['noise'][k:k + 1],
                seconds=bank['seconds'][k:k + 1], delays=bank['delays'][k:k + 1], levels=bank['levels'][k:k + 1], thresholds=bank['thresholds'], sos=bank['sos'])
    first, last = bank['mode_offsets'][k].min(), (bank['mode_offsets'][k] + bank['mode_counts'][k]).max()
    part.update({f'mode_{name}': bank[f'mode_{name}'][first:last] for name in ('frequency', 'amplitude', 'decay', 'phase')},
                mode_offsets=bank['mode_offsets'][k:k + 1] - first, mode_counts=bank['mode_counts'][k:k + 1])
    np.savez(Path(__file__).resolve().parent / 'reference_release.npz', **part)
    print('wrote reference_release.npz from', releases)
