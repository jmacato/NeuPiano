"""Fit the modal synth to every raw recording channel on the GPU and export the parameters the app plays."""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

from modal_piano import HOP, RATE, partial_frequencies, raw_recordings, third_octave_bins, tuning, window_size

BEATS = (0.0, .3, .8, 1.5, 3.0)
MAX_DECAY, MAX_BEAT = 60.0, 12.0


class Throttle:
    """Keeps synchronized Metal work at or below the given duty and memory, and waits out critical macOS memory pressure."""

    def __init__(self, duty, memory_gib):
        self.duty, self.limit = duty, int(memory_gib * 2 ** 30)
        torch.mps.set_per_process_memory_fraction(min(1., self.limit / torch.mps.recommended_max_memory()))
        self.busy = self.idle = 0.
        self.peak = self.chunks = 0
        self.started = time.monotonic()

    def pressure(self):
        while subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                             capture_output=True, text=True, check=True).stdout.strip() == '4':
            time.sleep(30)

    def run(self, work):
        if self.chunks % 20 == 0:
            self.pressure()
        start = time.monotonic()
        result = work()
        torch.mps.synchronize()
        elapsed = time.monotonic() - start
        self.peak = max(self.peak, torch.mps.driver_allocated_memory())
        rest = elapsed * (1 / self.duty - 1)
        time.sleep(rest)
        self.busy, self.idle, self.chunks = self.busy + elapsed, self.idle + rest, self.chunks + 1
        return result

    def report(self):
        return (f'GPU duty {self.busy / max(self.busy + self.idle, 1e-9):.0%}, peak memory {self.peak / 2 ** 20:.0f} MiB, '
                f'elapsed {time.monotonic() - self.started:.0f} s')


def band_edges():
    return [((bins[0] - .5) * RATE / 2048, (bins[-1] + .5) * RATE / 2048) for bins in third_octave_bins(2048, RATE)]


def noise_filters():
    """Three biquads per 1/3-octave band; the top band is a high-pass, padded with an identity section."""
    sections = []
    for low, high in band_edges():
        if high >= RATE / 2:
            sos = butter(3, low, 'highpass', fs=RATE, output='sos')
        else:
            sos = butter(3, [low, high], 'bandpass', fs=RATE, output='sos')
        sos = np.concatenate([sos, np.tile([1., 0, 0, 1, 0, 0], (3 - len(sos), 1))])
        sections.append(sos)
    return np.array(sections)


def band_matrix(size, device):
    hz = np.fft.rfftfreq(size, 1 / RATE)
    matrix = np.zeros((len(band_edges()), len(hz)), np.float32)
    for band, bins in enumerate(third_octave_bins(size, RATE)):
        matrix[band, bins] = 1
    return torch.tensor(matrix, device=device)


def power(audio, size, device):
    window = torch.hann_window(size, periodic=True, device=device)
    spectrum = torch.stft(audio, size, HOP, window=window, center=True, pad_mode='constant', return_complex=True)
    return spectrum.real ** 2 + spectrum.imag ** 2


def unit_power(size, device):
    unit = torch.tensor(np.cos(2 * np.pi * 1000 * np.arange(RATE) / RATE), dtype=torch.float32, device=device)
    return float(power(unit, size, device)[:, 10].sum())


def coupling_table(size):
    weights = np.hanning(size) ** 2
    grid = np.arange(0, MAX_BEAT + .01, .005)
    phase = np.exp(2j * np.pi * np.outer(grid, np.arange(size)) / RATE)
    return np.abs(phase @ weights) / weights.sum()


def interpolate(table, beat):
    position = beat / .005
    index = position.floor().clamp(0, len(table) - 2).long()
    fraction = position - index
    return table[index] * (1 - fraction) + table[index + 1] * fraction


def masked_slope(level, t, mask):
    weight = mask.float()
    count = weight.sum(-1, keepdim=True).clamp_min(1)
    mean_t = (weight * t).sum(-1, keepdim=True) / count
    mean_y = (weight * level).sum(-1, keepdim=True) / count
    covariance = (weight * (t - mean_t) * (level - mean_y)).sum(-1)
    variance = (weight * (t - mean_t) ** 2).sum(-1).clamp_min(1e-9)
    slope = covariance / variance
    return torch.where(weight.sum(-1) >= 4, (-slope / 2).clamp(1e-3, MAX_DECAY - 1e-3), torch.ones_like(slope))


def envelope(log_a1, log_a2, d1, d2, beat, phase, rho, t):
    return (torch.exp(2 * log_a1 - 2 * d1 * t) + torch.exp(2 * log_a2 - 2 * d2 * t)
            + 2 * rho * torch.exp(log_a1 + log_a2 - (d1 + d2) * t) * torch.cos(2 * np.pi * beat * t + phase))


def logit(x, top):
    x = (x / top).clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def fit_envelopes(energy, floor, weight, t, table, throttle, steps):
    """Adam over 5 beat starts per partial; returns (log_a1, log_a2, d1, d2, beat, phase) and the per-partial loss."""
    usable = (energy > floor * 3) & (weight > 0)
    first = torch.where(usable.any(-1), usable.float().argmax(-1), torch.zeros_like(usable[..., 0], dtype=torch.long))
    last = usable.shape[-1] - 1 - usable.flip(-1).float().argmax(-1)
    level = torch.log(energy.clamp_min(1e-30))
    early = usable & (t < (t[first] + 1.0)[..., None])
    late = usable & (t > (t[last] / 2)[..., None])
    peak = torch.sqrt(torch.where(usable, energy, torch.zeros_like(energy)).amax(-1).clamp_min(1e-30))
    starts = len(BEATS)
    shape = (starts,) + peak.shape
    beats = torch.tensor(BEATS, device=energy.device).view(-1, *([1] * peak.dim())).expand(shape)
    parameters = [torch.log(peak * .7).expand(shape).clone(), torch.log(peak * .4).expand(shape).clone(),
                  logit(masked_slope(level, t, early), MAX_DECAY).expand(shape).clone(),
                  logit(masked_slope(level, t, late), MAX_DECAY).expand(shape).clone(),
                  logit(beats.clamp_min(1e-3), MAX_BEAT).clone(), torch.zeros(shape, device=energy.device)]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.002)
    target = 10 * torch.log10(energy + floor)
    total = weight.sum(-1).clamp_min(1)

    def losses():
        log_a1, log_a2, u1, u2, v, phase = parameters
        d1, d2, beat = MAX_DECAY * torch.sigmoid(u1), MAX_DECAY * torch.sigmoid(u2), MAX_BEAT * torch.sigmoid(v)
        model = envelope(log_a1[..., None], log_a2[..., None], d1[..., None], d2[..., None], beat[..., None],
                         phase[..., None], interpolate(table, beat)[..., None], t)
        error = 10 * torch.log10(model.clamp_min(0) + floor) - target
        return (error ** 2 * weight).sum(-1) / total

    def chunk():
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            losses().sum().backward()
            optimizer.step()
            scheduler.step()

    for _ in range(steps // 25):
        throttle.run(chunk)
    with torch.no_grad():
        loss = losses()
        best = loss.argmin(0)
        log_a1, log_a2, u1, u2, v, phase = [p.gather(0, best[None]).squeeze(0) for p in parameters]
        result = torch.stack([log_a1, log_a2, MAX_DECAY * torch.sigmoid(u1), MAX_DECAY * torch.sigmoid(u2),
                              MAX_BEAT * torch.sigmoid(v), torch.remainder(phase + np.pi, 2 * np.pi) - np.pi], -1)
        exists = usable.sum(-1) >= 8
    return result, exists


def fit_noise(residual, floor, weight, t, throttle, steps):
    """Burst g*exp(-t/tau) plus a constant floor per band, on log band energy."""
    tail = (t > (weight * t).amax(-1, keepdim=True) - 3) & (weight > 0)
    late = torch.where(tail, residual, torch.zeros_like(residual))
    late_level = (late.sum(-1) / tail.float().sum(-1).clamp_min(1)).clamp_min(floor[..., 0])
    burst = (residual[..., :10].amax(-1) - late_level).clamp_min(floor[..., 0])
    parameters = [torch.log(burst).clone(), torch.full_like(burst, np.log(.05)), torch.log(late_level).clone()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=.05)
    target = 10 * torch.log10(residual + floor)
    total = weight.sum(-1).clamp_min(1)

    def losses():
        gain, log_tau, rest = parameters
        model = torch.exp(gain[..., None] - t / torch.exp(log_tau[..., None].clamp(np.log(1e-3), np.log(30.)))) + torch.exp(rest[..., None])
        return (((10 * torch.log10(model + floor) - target) ** 2) * weight).sum(-1) / total

    def chunk():
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            losses().sum().backward()
            optimizer.step()

    for _ in range(steps // 25):
        throttle.run(chunk)
    gain, log_tau, rest = [p.detach() for p in parameters]
    return torch.stack([torch.exp(gain), torch.exp(log_tau.clamp(np.log(1e-3), np.log(30.))), torch.exp(rest)], -1)


def synthesize(partials, exists, frequencies, frames, device, seed, block=4096, theta=None):
    """String pairs for one channel; per-block phases come from float64 so long notes keep their pitch in float32."""
    a1, a2 = torch.exp(partials[:, 0]) * exists, torch.exp(partials[:, 1]) * exists
    d1, d2, beat, phase = partials[:, 2], partials[:, 3], partials[:, 4], partials[:, 5]
    beat64 = beat.cpu().numpy().astype(np.float64)
    strings = np.stack([frequencies + beat64 / 2, frequencies - beat64 / 2])
    theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(frequencies)) if theta is None else theta
    offsets = np.stack([theta, theta - phase.cpu().numpy().astype(np.float64)])
    amplitudes, decays = torch.stack([a1, a2]), torch.stack([d1, d2])
    hz = torch.tensor(strings, dtype=torch.float32, device=device)
    r = torch.arange(block, device=device, dtype=torch.float32)
    output = torch.zeros(frames, device=device)
    starts = np.arange(0, frames, block)
    group = int(np.clip(8e6 // (2 * len(frequencies) * block), 1, 16))
    for first in range(0, len(starts), group):
        n0 = starts[first:first + group]
        cycles = np.remainder(strings[..., None] * n0 / RATE, 1)
        base = torch.tensor(2 * np.pi * cycles + offsets[..., None], dtype=torch.float32, device=device)
        times = (torch.tensor(n0, dtype=torch.float32, device=device)[:, None] + r) / RATE
        angle = base[..., None] + 2 * np.pi * hz[..., None, None] * (r / RATE)
        level = amplitudes[..., None, None] * torch.exp(-decays[..., None, None] * times)
        chunk = (level * torch.cos(angle)).sum((0, 1)).reshape(-1)
        end = min(frames, n0[0] + len(chunk))
        output[n0[0]:end] = chunk[:end - n0[0]]
    return output


def fit_key(note, cache, device, throttle, steps, noise_units):
    layers = raw_recordings(cache, note)
    f0, inharmonicity = tuning(layers[9].mean(1), note)
    channels = [layer[:, channel] for layer in layers for channel in range(2)]
    lengths = np.array([len(c) for c in channels])
    audio = torch.zeros(len(channels), lengths.max(), device=device)
    for index, channel in enumerate(channels):
        audio[index, :len(channel)] = torch.tensor(channel, dtype=torch.float32, device=device)
    size = window_size(f0)
    count = int(np.searchsorted(partial_frequencies(f0, inharmonicity, 4000), 19000))
    frequencies = partial_frequencies(f0, inharmonicity, count)
    hz = np.fft.rfftfreq(size, 1 / RATE)
    rows = np.zeros((count, len(hz)), np.float32)
    gaps = np.zeros((count, len(hz)), np.float32)
    for n, f in enumerate(frequencies):
        following = frequencies[n + 1] if n + 1 < count else f + f0
        width = max(min(f0, following - f) * .45, 2.5 * RATE / size)
        rows[n] = (hz > f - width) & (hz < f + width)
        gap = (hz >= f + width) & (hz <= following - width)
        if not gap.any():
            gap = (hz >= f + width) & (hz < f + width + 2 * RATE / size)
        gaps[n] = gap / gap.sum()
    rows_t, gaps_t = torch.tensor(rows, device=device), torch.tensor(gaps, device=device)
    calibration = unit_power(size, device)
    energies, floors = [], []
    for index in range(len(channels)):
        spectrum = power(audio[index], size, device)
        energies.append(rows_t @ spectrum / calibration)
        gap = gaps_t @ spectrum
        padded = torch.nn.functional.pad(gap[None], (12, 12), mode='replicate')[0]
        floors.append(padded.unfold(-1, 25, 1).median(-1).values * rows_t.sum(1, keepdim=True) / calibration + 1e-30)
    energy, floor = torch.stack(energies), torch.stack(floors)
    t = torch.arange(energy.shape[-1], device=device, dtype=torch.float32) * HOP / RATE
    weight = (t[None] < torch.tensor(lengths / RATE, device=device, dtype=torch.float32)[:, None]).float()[:, None]
    table = torch.tensor(coupling_table(size), dtype=torch.float32, device=device)
    budget = max(1, int(6e6 // (len(BEATS) * count * energy.shape[-1])))
    partials, exists = [], []
    for start in range(0, len(channels), budget):
        part = slice(start, start + budget)
        result, found = fit_envelopes(energy[part], floor[part], weight[part], t, table, throttle, steps)
        partials.append(result)
        exists.append(found)
    partials, exists = torch.cat(partials), torch.cat(exists)

    band_power_matrix = band_matrix(2048, device)
    targets, tonal_bands = [], []
    for index in range(len(channels)):
        targets.append(band_power_matrix @ power(audio[index], 2048, device))
        tonal = throttle.run(lambda: synthesize(partials[index], exists[index], frequencies, lengths.max(), device, index))
        tonal_bands.append(band_power_matrix @ power(tonal, 2048, device))
    targets, tonal_bands = torch.stack(targets), torch.stack(tonal_bands)
    frames = min(targets.shape[-1], energy.shape[-1])
    targets, tonal_bands, t2 = targets[..., :frames], tonal_bands[..., :frames], t[:frames]
    noise_floor = (targets.amax((1, 2)) * 1e-8).clamp_min(1e-30)[:, None, None].expand(-1, len(band_edges()), frames)
    noise = fit_noise((targets - tonal_bands).clamp_min(0), noise_floor, weight[..., :frames], t2, throttle, steps // 2)
    noise[..., 0] /= noise_units[None]
    noise[..., 2] /= noise_units[None]
    amplitudes = torch.exp(partials[..., :2])
    keep = exists & (amplitudes.amax(-1) > 1e-6)
    return {
        'f0': f0, 'inharmonicity': inharmonicity, 'frequencies': frequencies, 'lengths': lengths,
        'partials': torch.cat([amplitudes, partials[..., 2:]], -1).cpu().numpy(), 'keep': keep.cpu().numpy(),
        'noise': noise.cpu().numpy()}


def measure_noise_units(device):
    """Band energy that unit-variance white noise produces after each band's filters, in the fit's units."""
    rng = np.random.default_rng(7)
    white = rng.standard_normal(RATE * 10)
    matrix = band_matrix(2048, device)
    units = []
    for band, sos in enumerate(noise_filters()):
        filtered = torch.tensor(sosfilt(sos, white), dtype=torch.float32, device=device)
        units.append(float((matrix @ power(filtered, 2048, device))[band, 25:-25].mean()))
    return torch.tensor(units, device=device)


def home_bands(frequency):
    lows = np.array([low for low, _ in band_edges()])
    return np.clip(np.searchsorted(lows, frequency, side='right') - 1, 0, len(lows) - 1)


def render(model, note, layer, channel, seed=0, partials=True, noise=True, phases=True, gains=None, attack=48):
    """The exported model for one recording channel as the app plays it; gains are optional per-frame log gains (kind, band, frame)."""
    k = int(np.flatnonzero(model['notes'] == note)[0])
    frames = int(round(model['seconds'][k, layer] * RATE))
    t = np.arange(frames) / RATE
    centers = np.arange(0 if gains is None else gains.shape[-1]) * HOP / RATE
    curves = {}

    def curve(kind, band):
        if (kind, band) not in curves:
            curves[kind, band] = np.exp(np.interp(t, centers, gains[kind, band]))
        return curves[kind, band]

    rng = np.random.default_rng(seed)
    audio = np.zeros(frames)
    start, count = model['offsets'][k, layer, channel], model['counts'][k, layer, channel]
    bands = home_bands(model['frequency'][start:start + count])
    for p in range(start, start + count if partials else start):
        f, beat, theta = model['frequency'][p], model['beat'][p], model['theta'][p] if phases else 0.
        strings = model['amplitude1'][p] * np.exp(-model['decay1'][p] * t) * np.cos(2 * np.pi * (f + beat / 2) * t + theta)
        strings += model['amplitude2'][p] * np.exp(-model['decay2'][p] * t) * np.cos(2 * np.pi * (f - beat / 2) * t + theta - model['phase'][p])
        audio += strings if gains is None else strings * curve(0, bands[p - start])
    first, modes = model['mode_offsets'][k, layer, channel], model['mode_counts'][k, layer, channel] if partials else 0
    for m in range(first, first + modes):
        mode = model['mode_amplitude'][m] * np.exp(-model['mode_decay'][m] * t) * np.cos(2 * np.pi * model['mode_frequency'][m] * t + model['mode_phase'][m])
        audio += mode if gains is None else mode * curve(0, home_bands(model['mode_frequency'][m]))
    audio[:attack] *= .5 - .5 * np.cos(np.pi * np.arange(min(attack, frames)) / attack)
    for band, (knock, fast, ring, slow, rest, *rise) in enumerate(model['noise'][k, layer, channel] if noise else []):
        filtered = sosfilt(model['sos'][band], rng.standard_normal(frames))
        level = (knock * np.exp(-t / fast) + ring * np.exp(-t / slow) + rest) * (-np.expm1(-t / rise[0]) if rise else 1)
        audio += np.sqrt(level) * filtered * (1 if gains is None else curve(1, band))
    return audio


def score(path, notes, cache):
    from modal_piano import smoothed_band_error
    from modal_piano import band_error
    model = dict(np.load(path))
    for note in notes:
        layers = raw_recordings(cache, note)
        errors = np.array([[band_error(layers[layer][:, channel], render(model, note, layer, channel), RATE),
                            smoothed_band_error(layers[layer][:, channel], render(model, note, layer, channel))]
                           for layer in range(10) for channel in range(2)])
        print(f'note {note}: playable model band error {errors[:, 0].mean():.2f} dB (20 ms), {errors[:, 1].mean():.2f} dB (100 ms)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes', type=int, nargs='+', default=list(range(21, 109)))
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('banks/modal_bank.npz'))
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--duty', type=float, default=.5)
    parser.add_argument('--memory-gib', type=float, default=2.)
    parser.add_argument('--score', action='store_true', help='Score an existing export against the raw recordings instead of fitting')
    args = parser.parse_args()
    if args.score:
        score(args.output, args.notes, args.cache)
        return
    if not 0 < args.duty <= .5:
        raise ValueError('GPU duty must stay at or below 50%')
    device = torch.device('mps')
    throttle = Throttle(args.duty, args.memory_gib)
    units = measure_noise_units(device)
    keys = {}
    for note in args.notes:
        started = time.monotonic()
        keys[note] = fit_key(note, args.cache, device, throttle, args.steps, units)
        print(f'note {note}: f0 {keys[note]["f0"]:.2f} Hz, {int(keys[note]["keep"].sum(1).mean())} partials per channel, '
              f'{time.monotonic() - started:.0f} s; {throttle.report()}', flush=True)
    export(args.output, keys, args.cache)


def export(path, keys, cache):
    notes = sorted(keys)
    offsets = np.zeros((len(notes), 10, 2), np.int32)
    counts = np.zeros((len(notes), 10, 2), np.int32)
    rows, noise, seconds = [], np.zeros((len(notes), 10, 2, len(band_edges()), 3), np.float32), np.zeros((len(notes), 10), np.float32)
    position = 0
    for k, note in enumerate(notes):
        key = keys[note]
        for index in range(20):
            layer, channel = divmod(index, 2)
            kept = np.flatnonzero(key['keep'][index])
            offsets[k, layer, channel], counts[k, layer, channel] = position, len(kept)
            for p in kept:
                rows.append((key['frequencies'][p], *key['partials'][index, p]))
            position += len(kept)
            noise[k, layer, channel] = key['noise'][index]
            seconds[k, layer] = key['lengths'][index] / RATE
    rows = np.array(rows, np.float32)
    temporary = path.with_suffix('.partial.npz')
    np.savez(temporary, notes=np.array(notes, np.int32), offsets=offsets, counts=counts,
             frequency=rows[:, 0], amplitude1=rows[:, 1], amplitude2=rows[:, 2], decay1=rows[:, 3], decay2=rows[:, 4],
             beat=rows[:, 5], phase=rows[:, 6], noise=noise, seconds=seconds,
             thresholds=np.load(cache / 'thresholds.npy').astype(np.int32), sos=noise_filters(),
             f0=np.array([keys[n]['f0'] for n in notes], np.float32),
             inharmonicity=np.array([keys[n]['inharmonicity'] for n in notes], np.float32))
    temporary.replace(path)
    print(f'wrote {path}: {len(notes)} keys, {len(rows)} partials')


if __name__ == '__main__':
    main()
