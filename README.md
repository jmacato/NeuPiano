# NeuPiano

A modal piano synth in C#. Each note is a set of damped string partials, body modes and filtered band noise. A small neural net can correct the band levels at 50 Hz. It runs as a desktop app and as a macOS AUv2 instrument built with NativeAOT.

This repository holds code only. It does not include fitted banks or the recordings that they come from.

## Layout

- `src/NeuPiano.Engine`: the synth engine (modal voices, control net, release sounds, MIDI file playback).
- `src/NeuPiano.App`: an Avalonia desktop app with an on-screen keyboard.
- `src/NeuPiano.AudioUnit`: an AUv2 instrument (`aumu Pno1 NeuP`) that uses only P/Invokes.
- `src/NeuPiano.Tests`: engine tests against Python renders of a synthetic bank.
- `training`: Python scripts that fit a bank to recordings and train the control net.

## Build

Requires the .NET 10 SDK.

```
cd src
dotnet build NeuPiano.slnx
dotnet test NeuPiano.Tests/NeuPiano.Tests.csproj
NeuPiano.AudioUnit/build.sh install
```

The app and the AU load `banks/modal_bank.npz` from the repository root. They also load `control_net.npz` and `release_bank.npz` from the same folder if those files exist.

## Fit a bank

The scripts need numpy, scipy and torch. Run them from the repository root.

1. Put 48 kHz stereo recordings in `recordings/`. Each key needs `key_<note>.npz` with arrays `layer0` to `layer9` of shape (frames, 2). Add `thresholds.npy` with the highest MIDI velocity of each layer. You can add `levels.npy` with a playback level per key and layer.
2. Run `fit_modal_gpu.py`, `fit_soundboard.py`, `fit_partials.py`, `fit_attack.py` and `fit_noise.py` in that order.
3. Run `train_control_net.py prepare --data <file>`, then `train_control_net.py train --data <file>`.
4. For key-up sounds, put release recordings in `recordings/release/` and run `fit_release.py`.

To make new test fixtures, run `src/NeuPiano.Tests/synthetic_banks.py <dir>`, then `src/NeuPiano.Tests/make_modal_reference.py` with the three bank files.

## License

MIT. See `LICENSE`.
