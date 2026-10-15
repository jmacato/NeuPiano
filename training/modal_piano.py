"""Fit a physical modal synth (stiff-string partials as detuned string pairs, hammer noise, recording floor) to each raw piano recording channel and score it with the band metric."""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import stft

RATE, SECONDS = 48000, 14
HOP = RATE // 50
LOWER, UPPER = [-50, -50, 0, 0, 0, -np.pi], [5, 5, 60, 60, 12, np.pi]


def raw_recordings(cache, note):
    """The first 14 s of each velocity layer's 48 kHz stereo recording of one key, from cache/key_<note>.npz with arrays layer0 to layer9 of shape (frames, 2)."""
    with np.load(cache / f'key_{note}.npz') as data:
        return [np.asarray(data[f'layer{layer}'], np.float64) for layer in range(10)]


def third_octave_bins(size, sample_rate):
    edges = [20.0, 50.0]
    while edges[-1] * 2 ** (1 / 3) < 20000:
        edges.append(edges[-1] * 2 ** (1 / 3))
    edges.append(sample_rate / 2 + 1)
    bands = np.digitize(np.fft.rfftfreq(size, 1 / sample_rate), edges) - 1
    bands[bands >= len(edges) - 1] = -1
    return [np.flatnonzero(bands == band) for band in range(len(edges) - 1) if np.any(bands == band)]


def band_energies(audio, sample_rate, size=2048, hop=None):
    hop = hop or sample_rate // 50
    power = np.abs(stft(audio, sample_rate, nperseg=size, noverlap=size - hop)[2]) ** 2
    return np.stack([power[bins].sum(axis=0) for bins in third_octave_bins(size, sample_rate)])


def band_error(reference, generated, sample_rate):
    target = band_energies(reference, sample_rate)
    actual = band_energies(generated, sample_rate)
    floor = max(float(target.max()) * 1e-8, 1e-18)
    active = target > floor * 10
    return float(np.mean(np.abs(10 * np.log10(np.maximum(actual[active], floor) / target[active]))))


def window_size(f0):
    return 16384 if f0 < 60 else 8192 if f0 < 150 else 4096 if f0 < 600 else 2048


def partial_frequencies(f0, inharmonicity, count):
    n = np.arange(1, count + 1)
    return n * f0 * np.sqrt(1 + inharmonicity * n ** 2)


def tuning(audio, note):
    """Fundamental from its own peak within a semitone of nominal, stiffness from the median over the upper partials."""
    segment = audio[int(.05 * RATE):int(3 * RATE)]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), 1 << 22))
    hz = np.fft.rfftfreq(1 << 22, 1 / RATE)

    def peak(low, high):
        band = (hz > low) & (hz < high)
        return hz[band][np.argmax(spectrum[band])], spectrum[band].max()

    nominal = 440 * 2 ** ((note - 69) / 12)
    first, _ = peak(nominal * .94, nominal * 1.06)
    inharmonicity, estimates = 0.0, []
    for n in range(2, 40):
        guess = n * first * np.sqrt((1 + inharmonicity * n ** 2) / (1 + inharmonicity))
        if guess + .3 * first > 15000:
            break
        frequency, level = peak(guess - .3 * first, guess + .3 * first)
        if level > spectrum.max() * 1e-4:
            ratio = (frequency / (n * first)) ** 2
            estimates.append(max((ratio - 1) / (n ** 2 - ratio), 0))
            inharmonicity = float(np.median(estimates))
    return first / np.sqrt(1 + inharmonicity), inharmonicity


def correlation(window, beat):
    weights = window ** 2
    return np.abs(np.sum(weights * np.exp(2j * np.pi * beat * np.arange(len(window)) / RATE))) / np.sum(weights)


def envelope_model(params, t, coupling):
    log_a1, log_a2, d1, d2, beat, phase = params
    a1, a2 = np.exp(log_a1), np.exp(log_a2)
    rho = coupling(beat)
    return (a1 ** 2 * np.exp(-2 * d1 * t) + a2 ** 2 * np.exp(-2 * d2 * t)
            + 2 * rho * a1 * a2 * np.exp(-(d1 + d2) * t) * np.cos(2 * np.pi * beat * t + phase))


def fit_partial(energy, t, floor, coupling):
    """Two detuned strings: amplitudes, decay rates (1/s), beat frequency (Hz) and beat phase."""
    usable = energy > floor * 3
    if usable.sum() < 8:
        return None
    level = 10 * np.log10(energy + floor)
    early = usable & (t < t[usable][0] + 1.0)
    late = usable & (t > t[usable][-1] / 2)
    def slope(mask):
        if mask.sum() < 4:
            return 1.0
        return max(-np.polyfit(t[mask], np.log(energy[mask]), 1)[0] / 2, 1e-3)
    peak = np.sqrt(energy[usable].max())
    best = None
    for beat in (0.0, .3, .8, 1.5, 3.0):
        start = np.clip([np.log(peak * .7), np.log(peak * .4), slope(early), slope(late), beat, 0.0], LOWER, UPPER)
        def residual(p):
            return 10 * np.log10(np.maximum(envelope_model(p, t, coupling), 0) + floor) - level
        result = least_squares(residual, start, bounds=(LOWER, UPPER), max_nfev=200)
        if best is None or result.cost < best.cost:
            best = result
    return best.x


def analyze(take, f0, inharmonicity):
    size = window_size(f0)
    window = np.hanning(size)
    hz, times, spectrum = stft(take, RATE, 'hann', size, size - HOP, boundary='zeros')
    power = np.abs(spectrum) ** 2
    count = int(np.searchsorted(partial_frequencies(f0, inharmonicity, 4000), 19000))
    frequencies = partial_frequencies(f0, inharmonicity, count)
    unit = np.cos(2 * np.pi * 1000 * np.arange(RATE) / RATE)
    calibration = (np.abs(stft(unit, RATE, 'hann', size, size - HOP, boundary='zeros')[2]) ** 2)[:, 10].sum()
    partials, tracks = [], []
    for n, f in enumerate(frequencies):
        following = frequencies[n + 1] if n + 1 < count else f + f0
        width = max(min(f0, following - f) * .45, 2.5 * RATE / size)
        rows = (hz > f - width) & (hz < f + width)
        gap = (hz >= f + width) & (hz <= following - width)
        if not gap.any():
            gap = (hz >= f + width) & (hz < f + width + 2 * RATE / size)
        floor = median_filter(power[gap].mean(0), 25, mode='nearest') * rows.sum() / calibration + 1e-30
        energy = power[rows].sum(0) / calibration
        fit = fit_partial(energy, times, floor, lambda beat: correlation(window, beat))
        if fit is not None:
            partials.append((f, *fit))
            tracks.append((f, energy))
    return np.array(partials), tracks, times


def synthesize_partials(partials, frames, seed=0):
    t = np.arange(frames) / RATE
    audio = np.zeros(frames)
    rng = np.random.default_rng(seed)
    for f, log_a1, log_a2, d1, d2, beat, phase in partials:
        theta = rng.uniform(0, 2 * np.pi)
        audio += np.exp(log_a1) * np.exp(-d1 * t) * np.cos(2 * np.pi * (f + beat / 2) * t + theta)
        audio += np.exp(log_a2) * np.exp(-d2 * t) * np.cos(2 * np.pi * (f - beat / 2) * t + theta - phase)
    return audio


def controlled_partials(tracks, times, frames, seed=0):
    """One sinusoid per partial following its measured 50 Hz energy track, as a perfect control-rate network would drive it."""
    t = np.arange(frames) / RATE
    audio = np.zeros(frames)
    rng = np.random.default_rng(seed)
    for f, energy in tracks:
        audio += np.sqrt(np.interp(t, times, energy)) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return audio


def band_noise(residual_bands, floor, frames, seed=1, exact=False):
    """Per 1/3-octave band: hammer burst g*exp(-t/tau) plus a constant floor fitted to the residual band energy, or the residual itself when exact."""
    rng = np.random.default_rng(seed)
    t_frames = np.arange(residual_bands.shape[1]) * HOP / RATE
    t = np.arange(frames) / RATE
    spectrum = np.fft.rfft(rng.standard_normal(frames))
    hz = np.fft.rfftfreq(frames, 1 / RATE)
    noise = np.zeros(frames)
    params = []
    for band, bins in enumerate(third_octave_bins(2048, RATE)):
        target = residual_bands[band]
        level = 10 * np.log10(target + floor)
        late = max(np.median(target[t_frames > t_frames[-1] - 3]), floor)
        start = [np.log(max(target[:10].max() - late, floor)), np.log(.05), np.log(late)]
        def residual(p):
            return 10 * np.log10(np.exp(p[0]) * np.exp(-t_frames / np.exp(p[1])) + np.exp(p[2]) + floor) - level
        fit = least_squares(residual, start, max_nfev=200).x
        params.append(fit)
        mask = (hz >= (bins[0] - .5) * RATE / 2048) & (hz < (bins[-1] + .5) * RATE / 2048)
        component = np.fft.irfft(np.where(mask, spectrum, 0), frames)
        unit = band_energies(component[:RATE * 2], RATE)[band, 5:-5].mean()
        envelope = np.interp(t, t_frames, target) if exact else np.exp(fit[0]) * np.exp(-t / np.exp(fit[1])) + np.exp(fit[2])
        noise += component * np.sqrt(envelope / unit)
    return noise, np.array(params)


def smoothed_band_error(reference, generated, frames=5):
    target = uniform_filter1d(band_energies(reference, RATE), frames, axis=1)
    actual = uniform_filter1d(band_energies(generated, RATE), frames, axis=1)
    floor = max(float(target.max()) * 1e-8, 1e-18)
    active = target > floor * 10
    return float(np.mean(np.abs(10 * np.log10(np.maximum(actual[active], floor) / target[active]))))


def fit_take(take, f0, inharmonicity):
    partials, tracks, times = analyze(take, f0, inharmonicity)
    tonal = synthesize_partials(partials, len(take))
    target = band_energies(take, RATE)
    floor = max(float(target.max()) * 1e-8, 1e-18)
    noise, noise_params = band_noise(np.maximum(target - band_energies(tonal, RATE), 0), floor, len(take))
    controlled = controlled_partials(tracks, times, len(take))
    ideal_noise, _ = band_noise(np.maximum(target - band_energies(controlled, RATE), 0), floor, len(take), exact=True)
    return partials, noise_params, tonal, tonal + noise, controlled + ideal_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notes', type=int, nargs='+')
    parser.add_argument('--cache', type=Path, default=Path('recordings'))
    args = parser.parse_args()
    for note in args.notes:
        layers = raw_recordings(args.cache, note)
        f0, inharmonicity = tuning(layers[9].mean(1), note)
        rows = []
        for stereo in layers:
            for channel in stereo.T:
                partials, _, tonal, synth, ideal = fit_take(channel, f0, inharmonicity)
                rows.append((band_error(channel, synth, RATE), band_error(channel, tonal, RATE), band_error(channel, ideal, RATE),
                             smoothed_band_error(channel, synth), smoothed_band_error(channel, ideal)))
        rows = np.array(rows).reshape(10, 2, 5).mean(1)
        print(f'note {note}: f0 {f0:.2f} Hz, B {inharmonicity:.2e}, seconds {len(layers[9]) / RATE:.1f}, '
              f'synth alone per layer {np.round(rows[:, 0], 2).tolist()}, mean {rows[:, 0].mean():.2f} dB '
              f'(partials alone {rows[:, 1].mean():.2f} dB, perfect 50 Hz control {rows[:, 2].mean():.2f} dB); '
              f'100 ms metric: synth {rows[:, 3].mean():.2f} dB, perfect control {rows[:, 4].mean():.2f} dB', flush=True)


if __name__ == '__main__':
    main()
