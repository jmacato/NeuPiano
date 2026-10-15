"""Fit each recording channel's band noise as a fast knock decay, a slow ring decay and a floor under a rise from note-on, so that strings, modes and noise together match the recording's band energy, with the first 60 ms measured in 5 ms windows."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import sosfilt

from fit_modal_gpu import Throttle, band_edges, band_matrix, measure_noise_units, noise_filters, power, synthesize
from modal_piano import HOP, RATE, raw_recordings

LAYERS, ATTACK, BLOCK = 10, 48, 64
SHORT, STEP, EARLY, LOWEST, FINE, LONG = 256, 48, .06, 1000., 4096, 4
SCALE = 2 * (10 / np.log(10)) ** 2


def components(model, k, layer, channel):
    start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
    strings = [(model['frequency'][j] + model['beat'][j] / 2, model['amplitude1'][j], model['decay1'][j], model['theta'][j]) for j in range(start, start + count)]
    strings += [(model['frequency'][j] - model['beat'][j] / 2, model['amplitude2'][j], model['decay2'][j], model['theta'][j] - model['phase'][j])
                for j in range(start, start + count)]
    first, modes = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel]
    strings += [(model['mode_frequency'][j], model['mode_amplitude'][j], model['mode_decay'][j], model['mode_phase'][j]) for j in range(first, first + modes)]
    return np.array(strings, np.float64).reshape(-1, 4)


def tonal_audio(rows, frames, device):
    frequency, amplitude, decay, phase = rows.T
    with np.errstate(divide='ignore'):
        partials = np.stack([np.log(amplitude), np.full_like(amplitude, -np.inf), decay, decay, np.zeros_like(decay), np.zeros_like(decay)], 1)
    audio = synthesize(torch.tensor(partials, dtype=torch.float32, device=device), torch.ones(len(rows), device=device), frequency, frames, device, 0,
                       theta=phase)
    ramp = torch.ones(frames, device=device)
    ramp[:ATTACK] = .5 - .5 * torch.cos(np.pi * torch.arange(min(ATTACK, frames), device=device) / ATTACK)
    return audio * ramp


def short_matrix(device):
    hz = np.fft.rfftfreq(SHORT, 1 / RATE)
    return torch.tensor(np.array([(hz >= low) & (hz < high) & (low >= LOWEST) for low, high in band_edges()], np.float32), device=device)


def short_power(audio, device):
    window = torch.hann_window(SHORT, periodic=True, device=device)
    spectrum = torch.stft(audio[:FINE], SHORT, STEP, window=window, center=True, pad_mode='constant', return_complex=True)
    return (spectrum.real ** 2 + spectrum.imag ** 2)[:, :int(EARLY * RATE) // STEP + 1]


def around(x, onset, half, length):
    """x from half samples before the onset to half samples past length, zero outside the recording, so frames centred on the onset see the real lead-in."""
    return np.concatenate([np.zeros(half), x, np.zeros(length + half)])[onset:onset + length + 2 * half]


def framed(audio, size, hop, device):
    window = torch.hann_window(size, periodic=True, device=device)
    spectrum = torch.stft(audio, size, hop, window=window, center=False, return_complex=True)
    return spectrum.real ** 2 + spectrum.imag ** 2


def leakage(narrow, units, device):
    """Band energy that unit-variance white noise through each band's filters puts in every band, in long and short frames, per unit of its own long-frame energy."""
    white = np.random.default_rng(7).standard_normal(RATE * 10)
    matrix = band_matrix(2048, device)
    window = torch.hann_window(SHORT, periodic=True, device=device)
    long, short = [], []
    for sos in noise_filters():
        filtered = torch.tensor(sosfilt(sos, white), dtype=torch.float32, device=device)
        long.append((matrix @ power(filtered, 2048, device))[:, 25:-25].mean(-1))
        spectrum = torch.stft(filtered, SHORT, 4 * STEP, window=window, center=True, pad_mode='constant', return_complex=True)
        short.append((narrow @ (spectrum.real ** 2 + spectrum.imag ** 2))[:, 10:-10].mean(-1))
    return torch.stack(long, 1) / units, torch.stack(short, 1) / units


def frame_weights(size, hop, frames, device):
    """Maps a per-sample power envelope over the first FINE samples to centred Hann frame energies at unit gain."""
    window = (.5 - .5 * np.cos(2 * np.pi * np.arange(size) / size)) ** 2
    weights = np.zeros((frames, FINE))
    for c in range(frames):
        m = np.arange(FINE) - c * hop + size // 2
        inside = (m >= 0) & (m < size)
        weights[c, inside] = window[m[inside]]
    return torch.tensor(weights.T / window.sum(), dtype=torch.float32, device=device)


def engine_power(level):
    """Per-sample noise power over the first FINE samples as the engine renders it: amplitude ramps linearly over each BLOCK, from zero at note-on."""
    amplitude = torch.cat([torch.zeros_like(level[..., :1]), torch.sqrt(level[..., 1:])], -1)
    fraction = (torch.arange(BLOCK, device=level.device) + 1) / BLOCK
    return (amplitude[..., :-1, None] + (amplitude[..., 1:] - amplitude[..., :-1])[..., None] * fraction).flatten(-2) ** 2


def fit(target, tonal, early, floor, weight, t, throttle, steps):
    """(g1 exp(-t/tau1) + g2 exp(-t/tau2) + rest) (1 - exp(-t/rise)) per band, with tau1 <= tau2, so that tonal plus noise, with each band's leakage into its neighbours,
    matches the recording in long frames (log error) and in the short early frames (Itakura-Saito)."""
    short_target, short_tonal, long_leak, short_leak, long_weights, short_weights = early
    device = target.device
    residual = (target - tonal).clamp_min(0)
    tail = (t > (weight * t).amax(-1, keepdim=True) - 3) & (weight > 0)
    late = (torch.where(tail, residual, torch.zeros_like(residual)).sum(-1) / tail.float().sum(-1).clamp_min(1)).clamp_min(floor[..., 0])
    knock = (residual[..., 0] - late).clamp_min(floor[..., 0])
    ring = (residual[..., 2:10].mean(-1) - late).clamp_min(floor[..., 0])
    parameters = [torch.log(knock), torch.full_like(knock, np.log(.01)), torch.log(ring), torch.full_like(knock, np.log(.2)), torch.log(late), torch.full_like(knock, np.log(.001))]
    parameters = [p.clone().requires_grad_(True) for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=.05)
    measured = 10 * torch.log10(target + floor)
    total = weight.sum(-1).clamp_min(1)
    short_floor = (short_target.amax((1, 2)) * 1e-8).clamp_min(1e-30)[:, None, None]
    mask = (short_leak.sum(1) > 0).float()
    blocks = torch.arange(FINE // BLOCK + 1, device=device, dtype=torch.float32) * BLOCK / RATE

    def model():
        g1, u1, g2, u2, rest, u3 = parameters
        tau1 = torch.exp(u1.clamp(np.log(5e-4), np.log(.1)))
        tau2 = tau1 + torch.exp(u2.clamp(np.log(1e-3), np.log(30.)))
        rise = torch.exp(u3.clamp(np.log(5e-5), np.log(.02)))
        level = lambda time: ((torch.exp(g1[..., None] - time / tau1[..., None]) + torch.exp(g2[..., None] - time / tau2[..., None]) + torch.exp(rest[..., None])) *
                              -torch.expm1(-time / rise[..., None]))
        fine = engine_power(level(blocks))
        long = torch.einsum('ab,nbt->nat', long_leak, torch.cat([fine @ long_weights, level(t[LONG:])], -1))
        return long, torch.einsum('ab,nbt->nat', short_leak, fine @ short_weights), tau1, tau2, rise

    def chunk():
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            long, short, _, _, _ = model()
            error = ((((10 * torch.log10(tonal + long + floor) - measured) ** 2) * weight).sum(-1) / total)
            r = (short_target + short_floor) / (short_tonal + short + short_floor)
            (error + SCALE * (r - torch.log(r) - 1).mean(-1) * mask).sum().backward()
            optimizer.step()

    for _ in range(steps // 25):
        throttle.run(chunk)
    with torch.no_grad():
        _, _, tau1, tau2, rise = model()
        g1, _, g2, _, rest, _ = parameters
        return torch.stack([torch.exp(g1), tau1, torch.exp(g2), tau2, torch.exp(rest), rise], -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--duty', type=float, default=.5)
    parser.add_argument('--memory-gib', type=float, default=2.)
    args = parser.parse_args()
    if not 0 < args.duty <= .5:
        raise ValueError('GPU duty must stay at or below 50%')
    model = dict(np.load(args.model))
    device = torch.device('mps')
    throttle = Throttle(args.duty, args.memory_gib)
    units = measure_noise_units(device)
    matrix, narrow = band_matrix(2048, device), short_matrix(device)
    long_leak, short_leak = leakage(narrow, units, device)
    long_weights, short_weights = frame_weights(2048, HOP, LONG, device), frame_weights(SHORT, STEP, int(EARLY * RATE) // STEP + 1, device)
    noise = np.zeros(model['noise'].shape[:-1] + (6,), np.float32)
    started = time.monotonic()
    for k, note in enumerate(model['notes']):
        layers = raw_recordings(args.cache, int(note))
        targets, tonals, short_targets, short_tonals, lengths = [], [], [], [], []
        for layer in range(LAYERS):
            length = int(round(model['seconds'][k, layer] * RATE))
            for channel in range(2):
                def work():
                    x = layers[layer][:, channel]
                    onset = int(np.argmax(np.abs(x) > np.abs(x).max() * .05))
                    size = min(length, len(x) - onset)
                    rows = components(model, k, layer, channel)
                    synth = tonal_audio(rows, size, device) if len(rows) else torch.zeros(size, device=device)
                    audio, lead = (torch.tensor(around(x, onset, half, span), dtype=torch.float32, device=device) for half, span in ((1024, size), (SHORT // 2, FINE)))
                    return (matrix @ framed(audio, 2048, HOP, device), matrix @ power(synth, 2048, device),
                            narrow @ framed(lead, SHORT, STEP, device)[:, :int(EARLY * RATE) // STEP + 1], narrow @ short_power(synth, device))
                target, tonal, short_target, short_tonal = throttle.run(work)
                targets.append(target)
                tonals.append(tonal)
                short_targets.append(short_target)
                short_tonals.append(short_tonal)
                lengths.append(target.shape[1])
        frames = max(lengths)
        target = torch.stack([torch.nn.functional.pad(x, (0, frames - x.shape[1])) for x in targets])
        tonal = torch.stack([torch.nn.functional.pad(x, (0, frames - x.shape[1])) for x in tonals])
        t = torch.arange(frames, device=device, dtype=torch.float32) * HOP / RATE
        weight = (torch.arange(frames, device=device)[None] < torch.tensor(lengths, device=device)[:, None]).float()[:, None]
        floor = (target.amax((1, 2)) * 1e-8).clamp_min(1e-30)[:, None, None].expand(-1, target.shape[1], frames)
        early = (torch.stack(short_targets), torch.stack(short_tonals), long_leak, short_leak, long_weights, short_weights)
        fitted = fit(target, tonal, early, floor, weight, t, throttle, args.steps)
        for column in (0, 2, 4):
            fitted[..., column] /= units[None]
        noise[k] = fitted.reshape(LAYERS, 2, -1, 6).cpu().numpy()
        print(f'note {note}: noise fitted; {throttle.report()}', flush=True)
    model['noise'] = noise
    temporary = args.output.with_suffix('.partial.npz')
    np.savez(temporary, **model)
    temporary.replace(args.output)
    print(f'wrote {args.output} in {time.monotonic() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
