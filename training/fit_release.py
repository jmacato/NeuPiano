"""Fit the key-up release recordings with the note pipeline's steps (strings, body modes, attack modes, band noise) and export them as a release bank the modal engine plays when a damper falls."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfiltfilt

import fit_attack as fa
import fit_noise as fn
from fit_modal_gpu import band_matrix, fit_envelopes, measure_noise_units, noise_filters, power
from fit_partials import FRAMES, measure_channel
from fit_phases import phases
from fit_soundboard import fit_modes
from modal_piano import HOP, RATE

LAYERS = 7
NAMES = ('frequency', 'amplitude1', 'amplitude2', 'decay1', 'decay2', 'beat', 'phase', 'theta')
MODES = ('frequency', 'amplitude', 'decay', 'phase')


class Unthrottled:
    def run(self, work):
        return work()


def release_recordings(cache, note):
    """48 kHz release recordings of one key from cache/key_<note>.npz: layer<i> (frames, 2), gain<i> playback level and id<i>, equal for layers that share a recording."""
    with np.load(cache / f'key_{note}.npz') as data:
        return {layer: (np.asarray(data[f'layer{layer}'], np.float64), float(data[f'gain{layer}']), int(data[f'id{layer}']))
                for layer in range(LAYERS) if f'layer{layer}' in data.files}


def strings_audio(rows, theta, frames):
    t = np.arange(frames) / RATE
    if not len(rows):
        return np.zeros(frames)
    f, a1, a2, d1, d2, beat, phase = (rows[:, i, None] for i in range(7))
    return (a1 * np.exp(-d1 * t) * np.cos(2 * np.pi * (f + beat / 2) * t + theta[:, None])
            + a2 * np.exp(-d2 * t) * np.cos(2 * np.pi * (f - beat / 2) * t + theta[:, None] - phase)).sum(0)


def fit_channel(x, frequencies, f0, early):
    """Partials, modes and 6-term band noise for one channel that starts at its release onset."""
    units, matrix, narrow, long_leak, short_leak, long_weights, short_weights = early
    corrected, envelopes, floors, skip = measure_channel(x, frequencies, f0)
    length = min(FRAMES, int(len(x) / RATE * 50))
    weight = np.zeros((1, 1, FRAMES), np.float32)
    weight[0, 0, int(np.ceil(skip * 50)):length] = 1
    weight[0, 0] *= np.minimum(1, 1 / np.maximum(np.arange(FRAMES) * HOP / RATE, 1e-3))
    t = torch.arange(FRAMES, dtype=torch.float32) * HOP / RATE
    fitted, exists = fit_envelopes(torch.tensor(envelopes[None], dtype=torch.float32),
                                   torch.tensor(np.repeat(floors[:, None], FRAMES, 1)[None], dtype=torch.float32),
                                   torch.tensor(weight), t, torch.ones(2401), Unthrottled(), 1500)
    p, ok = fitted[0].numpy().astype(np.float64), exists[0].numpy()
    rows = np.stack([corrected, np.exp(p[:, 0]), np.exp(p[:, 1]), p[:, 2], p[:, 3], p[:, 4], p[:, 5]], 1)[ok]
    body = fit_modes(x, f0)
    theta = phases(x, rows, body) if len(rows) else np.zeros(0)
    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
    frames = len(x) - onset
    lead, span = int(fa.LEAD * RATE), int(fa.WINDOW * RATE)
    segment = np.concatenate([np.zeros(lead), x])[onset:onset + 2 * lead + span]
    tonal = strings_audio(rows, theta, frames) + (fa.modes_audio(body, frames) if len(body) else 0)
    overlap = min(len(segment) - lead, frames)
    segment[lead:lead + overlap] -= tonal[:overlap]
    low = sosfiltfilt(butter(8, fa.CEILING, 'lowpass', fs=RATE, output='sos'), segment)[lead:lead + span:fa.FACTOR]
    attack, _ = fa.pursue(low, *fa.dictionary())
    if f0 >= fa.LOWEST:
        segment[lead:] -= fa.modes_audio(attack, len(segment) - lead) if len(attack) else 0
        attack = np.concatenate([attack, fa.prompt(segment, frequencies, f0)])
    modes = np.concatenate([body, attack]) if len(attack) else body
    synth = torch.tensor(strings_audio(rows, theta, frames) + (fa.modes_audio(modes, frames) if len(modes) else 0), dtype=torch.float32)
    audio = torch.tensor(fn.around(x, onset, 1024, frames), dtype=torch.float32)
    lead_in = torch.tensor(fn.around(x, onset, fn.SHORT // 2, fn.FINE), dtype=torch.float32)
    target = (matrix @ fn.framed(audio, 2048, HOP, torch.device('cpu')))[None]
    tonal_bands = (matrix @ power(synth, 2048, torch.device('cpu')))[None]
    count = min(target.shape[-1], tonal_bands.shape[-1])
    target, tonal_bands = target[..., :count], tonal_bands[..., :count]
    short_target = (narrow @ fn.framed(lead_in, fn.SHORT, fn.STEP, torch.device('cpu'))[:, :int(fn.EARLY * RATE) // fn.STEP + 1])[None]
    short_tonal = (narrow @ fn.short_power(synth, torch.device('cpu')))[None]
    times = torch.arange(count, dtype=torch.float32) * HOP / RATE
    floor = (target.amax((1, 2)) * 1e-8).clamp_min(1e-30)[:, None, None].expand(-1, target.shape[1], count)
    noise = fn.fit(target, tonal_bands, (short_target, short_tonal, long_leak, short_leak, long_weights, short_weights), floor,
                   torch.ones(1, 1, count), times, Unthrottled(), 1000)[0].numpy()
    for column in (0, 2, 4):
        noise[:, column] /= units
    rows = np.concatenate([rows, theta[:, None]], 1) if len(rows) else np.zeros((0, 8))
    return rows, modes.reshape(-1, 4), noise, frames


def early_terms():
    cpu = torch.device('cpu')
    units = measure_noise_units(cpu)
    matrix, narrow = band_matrix(2048, cpu), fn.short_matrix(cpu)
    long_leak, short_leak = fn.leakage(narrow, units, cpu)
    return (units.numpy(), matrix, narrow, long_leak, short_leak, fn.frame_weights(2048, HOP, fn.LONG, cpu),
            fn.frame_weights(fn.SHORT, fn.STEP, int(fn.EARLY * RATE) // fn.STEP + 1, cpu))


def fit_key(job):
    model_path, cache, k, note = job
    torch.set_num_threads(1)
    model = np.load(model_path)
    early = early_terms()
    f0 = float(model['f0'][k])
    recordings = release_recordings(cache, note)
    fitted = {}
    result = {}
    for layer, (stereo, gain, recording) in recordings.items():
        if recording not in fitted:
            delay = min(int(np.argmax(np.abs(stereo[:, c]) > np.abs(stereo[:, c]).max() * .05)) for c in range(2))
            channels = []
            for channel in range(2):
                start, count = model['offsets'][k, 9, channel], model['counts'][k, 9, channel]
                frequencies = model['frequency'][start:start + count].astype(np.float64)
                channels.append(fit_channel(stereo[delay:, channel], frequencies, f0, early))
            fitted[recording] = (delay, channels)
        result[layer] = (gain, *fitted[recording])
    return k, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings/release'))
    parser.add_argument('--output', type=Path, default=Path('banks/release_bank.npz'))
    parser.add_argument('--processes', type=int, default=6)
    args = parser.parse_args()
    model = np.load(args.model)
    notes = [int(n) for n in model['notes']]
    shape = (len(notes), LAYERS, 2)
    offsets, counts, mode_offsets, mode_counts = (np.zeros(shape, np.int32) for _ in range(4))
    noise = np.zeros(shape + (len(noise_filters()), 6), np.float32)
    seconds, delays, levels = (np.zeros(shape[:2], np.float32) for _ in range(3))
    rows, modes = [], []
    started = time.monotonic()
    jobs = [(args.model, args.cache, k, note) for k, note in enumerate(notes) if (args.cache / f'key_{note}.npz').exists()]
    with Pool(args.processes) as pool:
        results = dict(pool.imap_unordered(fit_key, jobs))
    for k in range(len(notes)):
        for layer in range(LAYERS):
            if layer not in results.get(k, {}):
                offsets[k, layer] = sum(len(r) for r in rows)
                mode_offsets[k, layer] = sum(len(m) for m in modes)
                continue
            gain, delay, channels = results[k][layer]
            levels[k, layer], delays[k, layer] = gain, delay / RATE
            seconds[k, layer] = min(c[3] for c in channels) / RATE
            for channel, (r, m, n, _) in enumerate(channels):
                offsets[k, layer, channel], counts[k, layer, channel] = sum(len(x) for x in rows), len(r)
                mode_offsets[k, layer, channel], mode_counts[k, layer, channel] = sum(len(x) for x in modes), len(m)
                rows.append(r)
                modes.append(m)
                noise[k, layer, channel] = n
    rows, modes = np.concatenate(rows).astype(np.float32), np.concatenate(modes).astype(np.float32)
    arrays = {name: rows[:, i] for i, name in enumerate(NAMES)} | {f'mode_{name}': modes[:, i] for i, name in enumerate(MODES)}
    temporary = args.output.with_suffix('.partial.npz')
    np.savez(temporary, notes=np.array(notes, np.int32), offsets=offsets, counts=counts, mode_offsets=mode_offsets, mode_counts=mode_counts,
             noise=noise, seconds=seconds, delays=delays, levels=levels, thresholds=np.load(args.cache / 'thresholds.npy').astype(np.int32), sos=noise_filters(),
             f0=model['f0'], inharmonicity=model['inharmonicity'], **arrays)
    temporary.replace(args.output)
    print(f'wrote {args.output}: {len(rows)} partials, {len(modes)} modes, {sum(len(r) for r in results.values())} slots in {time.monotonic() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
