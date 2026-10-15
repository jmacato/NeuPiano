"""Train the 50 Hz control net that corrects the modal synth's band levels toward the raw recordings."""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.signal import sosfilt

from fit_modal_gpu import Throttle, band_edges, band_matrix, home_bands, power, render, synthesize
from fit_noise import BLOCK, FINE, LONG, engine_power, frame_weights
from fit_soundboard import mode_audio
from modal_piano import HOP, RATE, raw_recordings, smoothed_band_error

FRAMES, BANDS, LAYERS, SPREAD = 701, 26, 10, 9
OUTPUTS = 2 * 2 * BANDS
BOUND, ORIGIN, SPAN, ONSET = 5., .05, 14., .02
EVALUATION = (21, 33, 45, 57, 60, 69, 81, 93, 105, 108)


def tonal_audio(model, rows, frames, device, theta):
    with np.errstate(divide='ignore'):
        partials = np.stack([np.log(model['amplitude1'][rows]), np.log(model['amplitude2'][rows]), model['decay1'][rows],
                             model['decay2'][rows], model['beat'][rows], model['phase'][rows]], 1)
    return synthesize(torch.tensor(partials, device=device), torch.ones(len(partials), device=device),
                      model['frequency'][rows].astype(np.float64), frames, device, 0, theta=theta)


def onset(t, length):
    return (t / length).clamp(0, 1) if length > 0 else torch.ones_like(t)


def attack(frames, device):
    ramp = torch.ones(frames, device=device)
    ramp[:48] = .5 - .5 * torch.cos(np.pi * torch.arange(min(48, frames), device=device) / 48)
    return ramp


def prepare(model, cache, path, device, throttle):
    """Band power of every recording channel, and of the synth's partials grouped by home band in the bands around it, scaled per band so the groups add up to the whole synth's power, at 20 ms frames."""
    matrix = band_matrix(2048, device)
    shape = (len(model['notes']), LAYERS, 2, BANDS, FRAMES)
    target, tonal = np.zeros(shape, np.float32), np.zeros(shape[:-1] + (SPREAD, FRAMES), np.float32)
    frames = np.zeros(shape[:2], np.int32)
    lost = 0.
    placed = np.arange(BANDS)[:, None] + np.arange(SPREAD) - SPREAD // 2
    inside = (placed >= 0) & (placed < BANDS)
    for k, note in enumerate(model['notes']):
        started = time.monotonic()
        layers = raw_recordings(cache, note)
        for layer in range(LAYERS):
            length = int(round(model['seconds'][k, layer] * RATE))
            for channel in range(2):
                def work():
                    audio = torch.tensor(layers[layer][:length, channel], dtype=torch.float32, device=device)
                    start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
                    home = home_bands(model['frequency'][start:start + count])
                    first, modes = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
                    mode_home = home_bands(model['mode_frequency'][first:first + modes])
                    groups, total = {}, torch.zeros(length, device=device)
                    for c in np.union1d(home, mode_home):
                        rows = start + np.flatnonzero(home == c)
                        synth = (tonal_audio(model, rows, length, device, model['theta'][rows].astype(np.float64)) if len(rows)
                                 else torch.zeros(length, device=device))
                        chosen = first + np.flatnonzero(mode_home == c)
                        if len(chosen):
                            synth = synth + mode_audio(np.stack([model[f'mode_{name}'][chosen] for name in ('frequency', 'amplitude', 'decay', 'phase')], 1),
                                                       length, device)
                        synth = synth * attack(length, device)
                        total += synth
                        groups[c] = (matrix @ power(synth, 2048, device)).cpu().numpy()
                    return (matrix @ power(audio, 2048, device)).cpu().numpy(), groups, (matrix @ power(total, 2048, device)).cpu().numpy()
                bands, groups, full = throttle.run(work)
                count = bands.shape[1]
                target[k, layer, channel, :, :count] = bands
                for c, group in groups.items():
                    for d in range(SPREAD):
                        if 0 <= c + d - SPREAD // 2 < BANDS:
                            tonal[k, layer, channel, c, d, :count] = group[c + d - SPREAD // 2]
                    lost = max(lost, 1 - tonal[k, layer, channel, c].sum() / group.sum())
                block = tonal[k, layer, channel, :, :, :count]
                incoherent = np.zeros((BANDS, count), np.float32)
                np.add.at(incoherent, placed[inside], block[inside])
                block[inside] *= np.divide(full, incoherent, out=np.ones_like(incoherent), where=incoherent > 0)[placed[inside]]
                frames[k, layer] = count
        print(f'note {note}: prepared in {time.monotonic() - started:.0f} s, worst group power outside +-{SPREAD // 2} bands {lost:.1e}; '
              f'{throttle.report()}', flush=True)
    np.savez(path, target=target, tonal=tonal, frames=frames)


def leakage(model, device):
    """Band power that each noise band's filters put into every 1/3-octave band, for unit-variance white noise."""
    white = np.random.default_rng(7).standard_normal(RATE * 10)
    matrix = band_matrix(2048, device)
    rows = [(matrix @ power(torch.tensor(sosfilt(sos, white), dtype=torch.float32, device=device), 2048, device))[:, 25:-25].mean(1)
            for sos in model['sos']]
    return torch.stack(rows)


class Controls(torch.nn.Module):
    """Log gains per channel, kind (partials, noise) and band from a per-recording embedding and log-time features."""

    def __init__(self, keys, width, depth, embedding, harmonics, start=ONSET):
        super().__init__()
        self.keys, self.harmonics, self.start = keys, harmonics, start
        self.embedding = torch.nn.Parameter(torch.randn(keys, LAYERS, embedding) * .1)
        sizes = [embedding + 3 + 2 * harmonics] + [width] * depth
        self.hidden = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.output = torch.nn.Linear(width, OUTPUTS)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, key, layer, t):
        u = torch.log1p(t.clamp(0, SPAN) / ORIGIN)[:, None] / np.log1p(SPAN / ORIGIN)
        k = torch.arange(1, self.harmonics + 1, device=t.device) * np.pi
        timing = torch.cat([u, torch.sin(u * k), torch.cos(u * k)], -1)
        static = torch.cat([self.embedding[key, layer], key[:, None] / (self.keys - 1), layer[:, None] / (LAYERS - 1)], -1)
        x = torch.cat([static[:, None].expand(-1, len(t), -1), timing[None].expand(len(key), -1, -1)], -1)
        for linear in self.hidden:
            x = torch.nn.functional.silu(linear(x))
        return BOUND * torch.tanh(self.output(x) / BOUND) * onset(t, self.start)[None, :, None]


class FreeControls(torch.nn.Module):
    """One free log gain per frame, channel, kind and band for each recording: the bound for this parameterization."""

    def __init__(self, keys, chosen):
        super().__init__()
        self.register_buffer('row', torch.zeros(keys, dtype=torch.long, device=chosen.device).index_copy(
            0, chosen, torch.arange(len(chosen), device=chosen.device)))
        self.table = torch.nn.Parameter(torch.zeros(len(chosen), LAYERS, FRAMES, OUTPUTS, device=chosen.device))

    def forward(self, key, layer, t):
        return BOUND * torch.tanh(self.table[self.row[key], layer] / BOUND) * onset(t, ONSET)[None, :, None]


def gains(controls):
    """(takes, frames, outputs) to (takes, channel, kind, band, frames)."""
    return controls.view(controls.shape[0], controls.shape[1], 2, 2, BANDS).permute(0, 2, 3, 4, 1)


def noise_envelope(noise, t):
    level = noise[..., 0:1] * torch.exp(-t / noise[..., 1:2]) + noise[..., 2:3] * torch.exp(-t / noise[..., 3:4]) + noise[..., 4:5]
    return level * -torch.expm1(-t / noise[..., 5:6]) if noise.shape[-1] > 5 else level


def early_noise(noise):
    """Noise power in the first LONG frames as the engine ramps it per block, weighted by each frame's window."""
    blocks = torch.arange(FINE // BLOCK + 1, device=noise.device, dtype=torch.float32) * BLOCK / RATE
    weights = frame_weights(2048, HOP, LONG, noise.device)
    return torch.stack([engine_power(noise_envelope(n, blocks)) @ weights for n in noise])


def smoothing_index(frames):
    """Gather indices for a 5-frame mean with scipy's reflect edges at each recording's last frame."""
    i = torch.arange(-2, FRAMES + 2, device=frames.device)[None]
    n = frames[:, None]
    i = torch.where(i < 0, -i - 1, torch.where(i >= n, 2 * n - 1 - i, i))
    return i.clamp(0, FRAMES - 1)


def smooth(x, index):
    padded = torch.gather(x, -1, index[:, None, None].expand(*x.shape[:-1], -1))
    return padded.unfold(-1, 5, 1).mean(-1)


def take_errors(predicted, target, frames):
    """Per-channel 100 ms band error in dB; quiet cells only count when the prediction rises above them."""
    index = smoothing_index(frames)
    predicted = torch.cat([target[..., :1], predicted[..., 1:]], -1)
    target, predicted = smooth(target, index), smooth(predicted, index)
    valid = (torch.arange(FRAMES, device=frames.device)[None] < frames[:, None])[:, None, None]
    floor = (target * valid).amax((-2, -1), keepdim=True) * 1e-8
    active = (target > floor * 10) & valid
    error = (10 * torch.log10(predicted.clamp_min(floor) / target.clamp_min(floor))).abs()
    quiet = torch.relu(10 * torch.log10(predicted.clamp_min(floor) / (floor * 10))) * (valid & ~active)
    count = active.sum((-2, -1)).clamp_min(1)
    return (error * active).sum((-2, -1)) / count, quiet.sum((-2, -1)) / count


def predict(net, key, layer, data, leak):
    t = torch.arange(FRAMES, device=leak.device, dtype=torch.float32) * HOP / RATE
    g = gains(net(key, layer, t))
    weighted = torch.exp(2 * g[:, :, 0, :, None]) * data['tonal'][key.cpu(), layer.cpu()].to(leak.device)
    tonal = sum(torch.nn.functional.pad(weighted[:, :, :, d], (0, 0, d, SPREAD - 1 - d)) for d in range(SPREAD))
    tonal = tonal[:, :, SPREAD // 2:SPREAD // 2 + BANDS]
    noise = torch.exp(2 * g[:, :, 1]) * torch.cat([data['early'][key, layer], noise_envelope(data['noise'][key, layer], t)[..., LONG:]], -1)
    return tonal + torch.einsum('ncbf,bd->ncdf', noise, leak)


def fit(net, data, leak, keys, throttle, steps, batch, rate, label):
    """Adam on the mean in-model 100 ms band error over all channels of the given keys."""
    layers = torch.arange(LAYERS, device=leak.device)
    pairs = torch.cartesian_prod(keys, layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=rate / 100)
    history = []

    def chunk():
        for _ in range(20):
            chosen = pairs[torch.randperm(len(pairs), device=leak.device)[:batch]]
            key, layer = chosen[:, 0], chosen[:, 1]
            error, quiet = take_errors(predict(net, key, layer, data, leak), data['target'][key, layer], data['frames'][key, layer])
            loss = (error + quiet).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            history.append(float(error.detach().mean()))

    for step in range(0, steps, 20):
        throttle.run(chunk)
        if (step + 20) % 500 == 0:
            print(f'{label} step {step + 20}: in-model error {np.mean(history[-100:]):.3f} dB; {throttle.report()}', flush=True)
    return net


@torch.no_grad()
def in_model(net, data, leak, keys, throttle):
    rows = []
    for k in keys.tolist():
        key = torch.full((LAYERS,), k, device=leak.device)
        layer = torch.arange(LAYERS, device=leak.device)
        error, _ = throttle.run(lambda: take_errors(predict(net, key, layer, data, leak), data['target'][key, layer], data['frames'][key, layer]))
        rows.append(float(error.mean()))
    return np.array(rows)


@torch.no_grad()
def control_curves(net, k, device):
    t = torch.arange(FRAMES, device=device, dtype=torch.float32) * HOP / RATE
    key = torch.full((LAYERS,), k, device=device)
    return gains(net(key, torch.arange(LAYERS, device=device), t)).cpu().numpy().astype(np.float64)


def score_note(job):
    model_path, cache, note, curves = job
    model = dict(np.load(model_path))
    layers = raw_recordings(cache, note)
    rows = []
    for layer in range(LAYERS):
        for channel in range(2):
            reference = layers[layer][:, channel]
            alone = smoothed_band_error(reference, render(model, note, layer, channel))
            controlled = smoothed_band_error(reference, render(model, note, layer, channel, gains=curves[layer, channel]))
            rows.append((alone, controlled))
    return note, np.array(rows).mean(0)


def score(model_path, cache, curves, processes):
    """Real renders scored with the agreed metric: synth alone and synth plus controls, per key."""
    jobs = [(model_path, cache, note, c) for note, c in curves.items()]
    with Pool(processes) as pool:
        return dict(pool.map(score_note, jobs))


def export(net, path, notes, harmonics):
    arrays = {'embedding': net.embedding.detach().cpu().numpy(), 'lows': np.array([low for low, _ in band_edges()], np.float32),
              'scalars': np.array([harmonics, ORIGIN, SPAN, BOUND, len(net.hidden) + 1, notes[0], len(notes), net.start], np.float32)}
    for i, linear in enumerate(list(net.hidden) + [net.output]):
        arrays[f'weight{i}'] = linear.weight.detach().cpu().numpy()
        arrays[f'bias{i}'] = linear.bias.detach().cpu().numpy()
    np.savez(path, **arrays)


def load(path, device):
    stored = np.load(path)
    harmonics, _, _, _, layers, _, keys = stored['scalars'][:7]
    start = float(stored['scalars'][7]) if len(stored['scalars']) > 7 else 0.
    net = Controls(int(keys), stored['weight0'].shape[0], int(layers) - 1, stored['embedding'].shape[2], int(harmonics), start).to(device)
    with torch.no_grad():
        net.embedding.copy_(torch.tensor(stored['embedding']))
        for i, linear in enumerate(list(net.hidden) + [net.output]):
            linear.weight.copy_(torch.tensor(stored[f'weight{i}']))
            linear.bias.copy_(torch.tensor(stored[f'bias{i}']))
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'bound', 'train', 'score'])
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('banks/control_net.npz'))
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--rate', type=float, default=2e-3)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--embedding', type=int, default=32)
    parser.add_argument('--harmonics', type=int, default=16)
    parser.add_argument('--score-notes', type=int, nargs='*', default=list(EVALUATION))
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--duty', type=float, default=.5)
    parser.add_argument('--memory-gib', type=float, default=2.)
    args = parser.parse_args()
    if not 0 < args.duty <= .5:
        raise ValueError('GPU duty must stay at or below 50%')
    device = torch.device('mps')
    throttle = Throttle(args.duty, args.memory_gib)
    model = dict(np.load(args.model))
    if args.command == 'prepare':
        prepare(model, args.cache, args.data, device, throttle)
        return
    stored = np.load(args.data)
    data = {'target': torch.tensor(stored['target'], device=device), 'tonal': torch.tensor(stored['tonal'])}
    data['frames'] = torch.tensor(stored['frames'], device=device, dtype=torch.long)
    data['noise'] = torch.tensor(model['noise'], device=device)
    data['early'] = early_noise(data['noise'])
    leak = leakage(model, device)
    notes = list(model['notes'])
    if args.command == 'score':
        net = load(args.output, device)
        every = torch.arange(len(notes), device=device)
        for n, error in zip(notes, in_model(net, data, leak, every, throttle)):
            print(f'note {n}: in-model {error:.2f} dB', flush=True)
    elif args.command == 'bound':
        keys = torch.tensor([notes.index(n) for n in args.score_notes], device=device)
        net = FreeControls(len(notes), keys)
        fit(net, data, leak, keys, throttle, args.steps, args.batch, args.rate, 'free')
    else:
        keys = torch.arange(len(notes), device=device)
        net = Controls(len(notes), args.width, args.depth, args.embedding, args.harmonics).to(device)
        fit(net, data, leak, keys, throttle, args.steps, args.batch, args.rate, 'net')
        export(net, args.output, notes, args.harmonics)
        print(f'wrote {args.output}', flush=True)
    chosen = torch.tensor([notes.index(n) for n in args.score_notes], device=device)
    alone = in_model(lambda key, layer, t: torch.zeros(len(key), len(t), OUTPUTS, device=device), data, leak, chosen, throttle)
    controlled = in_model(net, data, leak, chosen, throttle)
    if not len(chosen):
        return
    curves = {n: control_curves(net, notes.index(n), device) for n in args.score_notes}
    real = score(args.model, args.cache, curves, args.processes)
    for i, n in enumerate(args.score_notes):
        print(f'note {n}: in-model {alone[i]:.2f} -> {controlled[i]:.2f} dB, rendered {real[n][0]:.2f} -> {real[n][1]:.2f} dB', flush=True)
    rendered = np.array([real[n] for n in args.score_notes])
    print(f'mean rendered {rendered[:, 0].mean():.2f} -> {rendered[:, 1].mean():.2f} dB, worst key {rendered[:, 1].max():.2f} dB', flush=True)


if __name__ == '__main__':
    main()
