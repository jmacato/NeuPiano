using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuPiano.Engine;

public sealed class ModalBank
{
    public const int Rate = 48000, Channels = 2, Sections = 3;

    ModalBank(Dictionary<string, NpyArray> arrays)
    {
        var notes = Ints(arrays["notes"]);
        FirstNote = notes[0];
        Keys = notes.Length;
        if (notes.Where((note, i) => note != FirstNote + i).Any())
            throw new InvalidDataException("Modal bank keys must be contiguous");
        Offsets = Ints(arrays["offsets"]);
        Counts = Ints(arrays["counts"]);
        Frequency = arrays["frequency"].ToFloats();
        Amplitude1 = arrays["amplitude1"].ToFloats();
        Amplitude2 = arrays["amplitude2"].ToFloats();
        Decay1 = arrays["decay1"].ToFloats();
        Decay2 = arrays["decay2"].ToFloats();
        Beat = arrays["beat"].ToFloats();
        Phase = arrays["phase"].ToFloats();
        Theta = arrays["theta"].ToFloats();
        Noise = arrays["noise"].ToFloats();
        Bands = arrays["noise"].Shape[3];
        NoiseTerms = arrays["noise"].Shape[4];
        Seconds = arrays["seconds"].ToFloats();
        Layers = arrays["seconds"].Shape[1];
        Delays = arrays.TryGetValue("delays", out var delays) ? delays.ToFloats() : new float[Seconds.Length];
        Thresholds = Ints(arrays["thresholds"]);
        Sos = arrays["sos"].ToDoubles();
        ModeOffsets = Ints(arrays["mode_offsets"]);
        ModeCounts = Ints(arrays["mode_counts"]);
        ModeFrequency = arrays["mode_frequency"].ToFloats();
        ModeAmplitude = arrays["mode_amplitude"].ToFloats();
        ModeDecay = arrays["mode_decay"].ToFloats();
        ModePhase = arrays["mode_phase"].ToFloats();
        Levels = arrays["levels"].ToFloats();
        var slots = Keys * Layers * Channels;
        if (Offsets.Length != slots || Counts.Length != slots || NoiseTerms is < 5 or > 6 || Noise.Length != slots * Bands * NoiseTerms || Seconds.Length != Keys * Layers ||
            Thresholds.Length != Layers || Sos.Length != Bands * Sections * 6 || Offsets.Zip(Counts).Any(r => r.First + r.Second > Frequency.Length) ||
            ModeOffsets.Length != slots || ModeCounts.Length != slots || Levels.Length != Keys * Layers || Delays.Length != Keys * Layers || Theta.Length != Frequency.Length ||
            ModeOffsets.Zip(ModeCounts).Any(r => r.First + r.Second > ModeFrequency.Length))
            throw new InvalidDataException("Modal bank arrays do not match its key, layer and band counts");
    }

    public static ModalBank Load(string path) => new(Npz.Load(path));

    static int[] Ints(NpyArray array) => Array.ConvertAll(array.ToDoubles(), v => (int)v);

    public int FirstNote { get; }
    public int Keys { get; }
    public int Layers { get; }
    public int Bands { get; }
    public int NoiseTerms { get; }
    public int Partials => Frequency.Length;
    public int[] Thresholds { get; }
    internal int[] Offsets { get; }
    internal int[] Counts { get; }
    internal float[] Frequency { get; }
    internal float[] Amplitude1 { get; }
    internal float[] Amplitude2 { get; }
    internal float[] Decay1 { get; }
    internal float[] Decay2 { get; }
    internal float[] Beat { get; }
    internal float[] Phase { get; }
    internal float[] Theta { get; }
    internal float[] Noise { get; }
    internal float[] Seconds { get; }
    internal float[] Delays { get; }
    internal double[] Sos { get; }
    internal int[] ModeOffsets { get; }
    internal int[] ModeCounts { get; }
    internal float[] ModeFrequency { get; }
    internal float[] ModeAmplitude { get; }
    internal float[] ModeDecay { get; }
    internal float[] ModePhase { get; }
    internal float[] Levels { get; }

    public bool Contains(int note) => note >= FirstNote && note < FirstNote + Keys;

    public int Layer(int velocity)
    {
        for (var i = 0; i < Layers; i++)
            if (velocity <= Thresholds[i])
                return i;
        return Layers - 1;
    }

    internal int Slot(int key, int layer, int channel) => (key * Layers + layer) * Channels + channel;
}

public sealed class ModalPiano
{
    public const int MaxVoices = 64, Block = 64, Resync = 4096, FadeFrames = 24000, AttackFrames = 48, UndampedFrom = 91, HoldLayers = 4;
    public const float DynamicRangeDb = 30, DampedDbPerSecond = 100, UndampedDbPerSecond = 2;
    const double Cutoff = 4e-6;
    static readonly float DamperFactor = Factor(DampedDbPerSecond), UndampedFactor = Factor(UndampedDbPerSecond);
    static readonly float RestrikeFactor = MathF.Exp(-1f / (.03f * ModalBank.Rate));

    readonly ModalBank bank;
    readonly ModalBank? releases;
    readonly ControlNet? controls;
    readonly List<Voice> voices = [], pending = [];
    readonly Voice?[] latest = new Voice?[128];
    float dynamicRangeDb = DynamicRangeDb, releaseExponent = 1, pitch = 1;
    int transpose, polyphony = MaxVoices;
    byte[]? velocityMap;
    readonly Band[] bands;
    readonly double[] energy;
    readonly float[] gains = new float[Block], sum = new float[Block];
    bool pedal;

    public ModalPiano(ModalBank bank, ControlNet? controls = null, ModalBank? releases = null)
    {
        if (controls is not null && (controls.Bands != bank.Bands || !controls.Contains(bank.FirstNote) || !controls.Contains(bank.FirstNote + bank.Keys - 1)))
            throw new ArgumentException("The control net does not cover this modal bank", nameof(controls));
        if (releases is not null && (releases.Bands != bank.Bands || releases.Layers <= HoldLayers))
            throw new ArgumentException("The release bank does not match this modal bank", nameof(releases));
        this.bank = bank;
        this.controls = controls;
        this.releases = releases;
        bands = Enumerable.Range(0, bank.Bands * ModalBank.Channels)
            .Select(i => new Band(bank, i % bank.Bands, i / bank.Bands, (uint)(i * 7919 + 1))).ToArray();
        energy = new double[bands.Length];
    }

    public ModalBank Bank => bank;
    public bool HasControls => controls is not null;
    public bool ControlsOn { get; set; } = true;
    public bool HasReleases => releases is not null;
    public bool ReleasesOn { get; set; } = true;
    public int ActiveVoices => voices.Count;
    public bool PartialsOn { get; init; } = true;
    public bool NoiseOn { get; init; } = true;

    public void Apply(PianoSettings settings)
    {
        dynamicRangeDb = settings.DynamicRange;
        releaseExponent = 1 / settings.ReleaseTime;
        ReleasesOn = settings.Releases;
        polyphony = Math.Clamp(settings.Voices, 4, MaxVoices);
        transpose = settings.Transpose + 12 * settings.Octave;
        pitch = (float)(Math.Pow(2, settings.Tune / 1200.0) * settings.A440 / 440);
        velocityMap = settings.VelocityTable();
    }

    public void NoteOn(int note, int velocity)
    {
        if (velocity == 0)
        {
            NoteOff(note);
            return;
        }
        var key = note + transpose;
        if (!bank.Contains(key))
            return;
        foreach (var sounding in voices)
            if (sounding.Note == note)
                sounding.Release(RestrikeFactor);
        while (voices.Count >= polyphony)
            voices.RemoveAt(Quietest());
        velocity = Math.Max(1, (int)(velocityMap?[velocity & 0x7F] ?? Math.Min(velocity, 127)));
        var layer = bank.Layer(velocity);
        var gain = bank.Levels[(key - bank.FirstNote) * bank.Layers + layer] * MathF.Pow(10, -dynamicRangeDb / 20 * (127 - velocity) / 127f);
        var voice = new Voice(this, bank, note, key, layer, gain, ControlsOn ? controls : null, pitch) { Velocity = velocity };
        voices.Add(voice);
        latest[note] = voice;
    }

    public void NoteOff(int note)
    {
        foreach (var voice in voices)
            if (voice.Note == note && voice.Held)
            {
                voice.Held = false;
                if (!pedal)
                    Damp(voice);
            }
        Flush();
    }

    public void Controller(int number, int value)
    {
        switch (number)
        {
            case 64:
                pedal = value >= 64;
                if (!pedal)
                    foreach (var voice in voices)
                        if (!voice.Held)
                            Damp(voice);
                Flush();
                break;
            case 120:
                voices.Clear();
                Array.Clear(latest);
                break;
            case 123:
                pedal = false;
                foreach (var voice in voices)
                {
                    voice.Held = false;
                    voice.Release(DamperFactor);
                }
                break;
        }
    }

    public static float VelocityGain(int velocity) => MathF.Pow(10, -DynamicRangeDb / 20 * (127 - velocity) / 127f);

    static float Factor(float decibelsPerSecond) => MathF.Pow(10, -decibelsPerSecond / 20 / ModalBank.Rate);

    internal static float DamperRate(int note) => note < UndampedFrom ? DamperFactor : UndampedFactor;

    public static int ReleaseLayer(int[] thresholds, double milliseconds, int velocity)
    {
        var layer = Pick(thresholds.AsSpan(0, HoldLayers), milliseconds);
        return layer == 0 || layer == 1 && velocity <= thresholds[HoldLayers] ? HoldLayers + Pick(thresholds.AsSpan(HoldLayers), velocity) : layer;
    }

    static int Pick(ReadOnlySpan<int> thresholds, double value)
    {
        for (var i = 0; i < thresholds.Length - 1; i++)
            if (value <= thresholds[i])
                return i;
        return thresholds.Length - 1;
    }

    void Damp(Voice voice)
    {
        if (voice.IsRelease || voice.Damped)
            return;
        voice.Damped = true;
        voice.Release(MathF.Pow(DamperRate(voice.Key), releaseExponent));
        if (releases is null || !ReleasesOn || latest[voice.Note] != voice || !releases.Contains(voice.Key))
            return;
        var layer = ReleaseLayer(releases.Thresholds, voice.Position * 1000.0 / ModalBank.Rate, voice.Velocity);
        var slot = (voice.Key - releases.FirstNote) * releases.Layers + layer;
        if (releases.Seconds[slot] > 0)
            pending.Add(new Voice(this, releases, voice.Note, voice.Key, layer, voice.Gain * releases.Levels[slot], null, pitch)
                { Held = false, IsRelease = true });
    }

    void Flush()
    {
        foreach (var voice in pending)
        {
            while (voices.Count >= polyphony)
                voices.RemoveAt(Quietest());
            voices.Add(voice);
        }
        pending.Clear();
    }

    int Quietest()
    {
        var quietest = 0;
        for (var i = 1; i < voices.Count; i++)
            if (voices[i].Level < voices[quietest].Level)
                quietest = i;
        return quietest;
    }

    public void Render(float[] stereo, int offset, int frames)
    {
        for (var done = 0; done < frames;)
        {
            var count = Math.Min(Block, frames - done);
            var start = offset + done;
            Array.Clear(energy);
            for (var i = voices.Count - 1; i >= 0; i--)
            {
                var voice = voices[i];
                voice.Render(stereo, start, count, gains, sum);
                voice.AddNoise(energy);
                if (voice.Finished)
                    voices.RemoveAt(i);
            }
            if (NoiseOn)
                for (var b = 0; b < bands.Length; b++)
                    bands[b].Render(stereo, start, count, energy[b]);
            done += count;
        }
    }

    sealed class Band
    {
        readonly double[] sos;
        readonly double[] state = new double[2 * ModalBank.Sections];
        readonly int channel;
        uint random;
        float amplitude;

        public Band(ModalBank bank, int band, int channel, uint seed)
        {
            sos = bank.Sos.AsSpan(band * ModalBank.Sections * 6, ModalBank.Sections * 6).ToArray();
            this.channel = channel;
            random = seed;
        }

        public void Render(float[] stereo, int start, int count, double energy)
        {
            var target = (float)Math.Sqrt(energy);
            if (target == 0 && amplitude == 0)
                return;
            var step = (target - amplitude) / count;
            for (var n = 0; n < count; n++)
            {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                var x = (random * (1.0 / 4294967296.0) - .5) * 3.4641016151377544;
                for (var s = 0; s < ModalBank.Sections; s++)
                {
                    var k = 6 * s;
                    var y = sos[k] * x + state[2 * s];
                    state[2 * s] = sos[k + 1] * x - sos[k + 4] * y + state[2 * s + 1];
                    state[2 * s + 1] = sos[k + 2] * x - sos[k + 5] * y;
                    x = y;
                }
                amplitude += step;
                stereo[2 * (start + n) + channel] += (float)(x * amplitude);
            }
            amplitude = target;
        }
    }

    sealed class Voice
    {
        readonly ModalPiano piano;
        readonly ModalBank bank;
        readonly Oscillators[] channels = new Oscillators[ModalBank.Channels];
        readonly ControlNet? net;
        readonly float[] current = [], next = [];
        readonly int noise, length, layer;
        int position, frame, waiting;
        float envelope = 1, decay = 1;

        public Voice(ModalPiano piano, ModalBank bank, int note, int played, int layer, float gain, ControlNet? controls, float pitch)
        {
            this.piano = piano;
            this.bank = bank;
            this.layer = layer;
            var key = played - bank.FirstNote;
            Note = note;
            Key = played;
            Gain = envelope = gain;
            length = (int)(bank.Seconds[key * bank.Layers + layer] * ModalBank.Rate);
            waiting = (int)(bank.Delays[key * bank.Layers + layer] * ModalBank.Rate);
            noise = bank.Slot(key, layer, 0) * bank.Bands * bank.NoiseTerms;
            if (controls is not null)
            {
                net = controls;
                (current, next) = (new float[net.Outputs], new float[net.Outputs]);
                net.Evaluate(played, layer, 0, current);
                net.Evaluate(played, layer, 1, next);
            }
            for (var c = 0; c < ModalBank.Channels; c++)
                channels[c] = new Oscillators(bank, bank.Slot(key, layer, c), net, current, next, c * ControlNet.Kinds * bank.Bands, pitch);
        }

        public int Note { get; }
        public int Key { get; }
        public int Velocity { get; init; }
        public float Gain { get; }
        public int Position => position;
        public bool IsRelease { get; init; }
        public bool Damped { get; set; }
        public bool Held { get; set; } = true;
        public bool Finished => position >= length + FadeFrames || envelope < 1e-4f;
        public float Level => envelope * Fade(position);

        float Fade(int at) => Math.Clamp((float)(length + FadeFrames - at) / FadeFrames, 0, 1);

        static float Attack(int at) => at >= AttackFrames ? 1 : .5f - .5f * MathF.Cos(MathF.PI * at / AttackFrames);

        public void Release(float factor) => decay = MathF.Min(decay, factor);

        public void Render(float[] stereo, int start, int count, float[] gains, float[] sum)
        {
            var skip = Math.Min(waiting, count);
            waiting -= skip;
            for (var done = skip; done < count;)
            {
                var run = count - done;
                if (net is not null)
                {
                    if (position == (frame + 1) * ControlNet.Frame)
                        Advance();
                    run = Math.Min(run, (frame + 1) * ControlNet.Frame - position);
                }
                Run(stereo, start + done, run, gains, sum);
                done += run;
            }
        }

        void Advance()
        {
            frame++;
            next.CopyTo(current, 0);
            if (net!.Changes(frame + 1))
                net.Evaluate(Key, layer, frame + 1, next);
        }

        void Run(float[] stereo, int start, int count, float[] gains, float[] sum)
        {
            for (var n = 0; n < count; n++)
            {
                gains[n] = envelope * Fade(position + n) * Attack(position + n);
                envelope *= decay;
            }
            if (piano.PartialsOn)
                for (var c = 0; c < ModalBank.Channels; c++)
                {
                    channels[c].Render(sum, count, position);
                    for (var n = 0; n < count; n++)
                        stereo[2 * (start + n) + c] += sum[n] * gains[n];
                }
            position += count;
        }

        public void AddNoise(double[] energy)
        {
            var gain = (double)envelope * Fade(position);
            if (gain == 0 || waiting > 0)
                return;
            var t = position / (double)ModalBank.Rate;
            var fraction = (position - frame * ControlNet.Frame) / (float)ControlNet.Frame;
            for (var b = 0; b < energy.Length; b++)
            {
                var k = noise + bank.NoiseTerms * b;
                var level = bank.Noise[k] * Math.Exp(-t / bank.Noise[k + 1]) + bank.Noise[k + 2] * Math.Exp(-t / bank.Noise[k + 3]) + bank.Noise[k + 4];
                if (bank.NoiseTerms > 5)
                    level *= 1 - Math.Exp(-t / bank.Noise[k + 5]);
                if (net is not null)
                {
                    var o = ((b / bank.Bands) * ControlNet.Kinds + 1) * bank.Bands + b % bank.Bands;
                    level *= Math.Exp(2 * (current[o] + (next[o] - current[o]) * fraction));
                }
                energy[b] += gain * gain * level;
            }
        }
    }

    sealed class Oscillators
    {
        readonly float[] amplitude, decay, re, im, wr, wi, baseR, baseI;
        readonly double[] frequency, phase;
        readonly int[] band;
        readonly float[] current, next;
        readonly int offset;
        readonly bool controlled;
        readonly double cutoff;
        int active;
        bool synced, sloped;

        public Oscillators(ModalBank bank, int slot, ControlNet? net, float[] current, float[] next, int offset, float pitch)
        {
            var first = bank.Offsets[slot];
            var count = bank.Counts[slot];
            var modes = bank.ModeCounts[slot];
            var lanes = Vector<float>.Count;
            var size = (2 * count + modes + lanes - 1) / lanes * lanes;
            (amplitude, decay, re, im, wr, wi) = (new float[size], new float[size], new float[size], new float[size], new float[size], new float[size]);
            (frequency, phase, band) = (new double[size], new double[size], new int[size]);
            (this.current, this.next, this.offset, controlled) = (current, next, offset, net is not null);
            for (var p = 0; p < count; p++)
            {
                var j = first + p;
                var theta = bank.Theta[j];
                Set(2 * p, bank.Amplitude1[j], bank.Decay1[j], (bank.Frequency[j] + bank.Beat[j] / 2) * pitch, theta);
                Set(2 * p + 1, bank.Amplitude2[j], bank.Decay2[j], (bank.Frequency[j] - bank.Beat[j] / 2) * pitch, theta - bank.Phase[j]);
                band[2 * p] = band[2 * p + 1] = offset + (net?.Band(bank.Frequency[j]) ?? 0);
            }
            for (var m = 0; m < modes; m++)
            {
                var j = bank.ModeOffsets[slot] + m;
                Set(2 * count + m, bank.ModeAmplitude[j], bank.ModeDecay[j], bank.ModeFrequency[j] * pitch, bank.ModePhase[j]);
                band[2 * count + m] = offset + (net?.Band(bank.ModeFrequency[j]) ?? 0);
            }
            (baseR, baseI) = ((float[])wr.Clone(), (float[])wi.Clone());
            active = 2 * count + modes;
            cutoff = Cutoff * (active > 0 ? amplitude.Max() : 0);
        }

        void Set(int i, float a, float d, float f, double offset)
        {
            amplitude[i] = a;
            decay[i] = d;
            frequency[i] = f;
            phase[i] = offset;
            var g = Math.Exp(-d / (double)ModalBank.Rate);
            var w = 2 * Math.PI * f / ModalBank.Rate;
            wr[i] = (float)(g * Math.Cos(w));
            wi[i] = (float)(g * Math.Sin(w));
        }

        public void Render(float[] sum, int count, int position)
        {
            for (var n = 0; n < count;)
            {
                var at = position + n;
                if (controlled && (!sloped || at % ControlNet.Frame == 0))
                    Slope();
                if (!synced || at % Resync == 0)
                    Sync(at);
                var run = Math.Min(count - n, Resync - at % Resync);
                if (controlled)
                    run = Math.Min(run, ControlNet.Frame - at % ControlNet.Frame);
                Step(sum.AsSpan(n, run));
                n += run;
            }
        }

        void Slope()
        {
            sloped = true;
            for (var j = 0; j < active; j++)
            {
                var growth = MathF.Exp((next[band[j]] - current[band[j]]) / ControlNet.Frame);
                wr[j] = baseR[j] * growth;
                wi[j] = baseI[j] * growth;
            }
        }

        double Gain(int j, int at) =>
            controlled ? Math.Exp(current[band[j]] + (next[band[j]] - current[band[j]]) * (at % ControlNet.Frame) / (double)ControlNet.Frame) : 1;

        void Sync(int at)
        {
            synced = true;
            var t = at / (double)ModalBank.Rate;
            for (var j = active - 1; j >= 0; j--)
            {
                var magnitude = amplitude[j] * Math.Exp(-decay[j] * t) * Gain(j, at);
                if (magnitude < cutoff)
                {
                    Remove(j);
                    continue;
                }
                var cycles = frequency[j] * at / ModalBank.Rate;
                var angle = 2 * Math.PI * (cycles - Math.Floor(cycles)) + phase[j];
                re[j] = (float)(magnitude * Math.Cos(angle));
                im[j] = (float)(magnitude * Math.Sin(angle));
            }
            Array.Clear(re, active, re.Length - active);
            Array.Clear(im, active, im.Length - active);
        }

        void Remove(int j)
        {
            var last = --active;
            (amplitude[j], decay[j], frequency[j], phase[j], band[j]) = (amplitude[last], decay[last], frequency[last], phase[last], band[last]);
            (re[j], im[j], wr[j], wi[j], baseR[j], baseI[j]) = (re[last], im[last], wr[last], wi[last], baseR[last], baseI[last]);
        }

        void Step(Span<float> output)
        {
            var lanes = Vector<float>.Count;
            var groups = (active + lanes - 1) / lanes;
            var r = MemoryMarshal.Cast<float, Vector<float>>(re.AsSpan(0, groups * lanes));
            var i = MemoryMarshal.Cast<float, Vector<float>>(im.AsSpan(0, groups * lanes));
            var a = MemoryMarshal.Cast<float, Vector<float>>(wr.AsSpan(0, groups * lanes));
            var b = MemoryMarshal.Cast<float, Vector<float>>(wi.AsSpan(0, groups * lanes));
            for (var n = 0; n < output.Length; n++)
            {
                var acc = Vector<float>.Zero;
                for (var g = 0; g < groups; g++)
                {
                    var x = r[g];
                    var y = i[g];
                    acc += x;
                    r[g] = x * a[g] - y * b[g];
                    i[g] = x * b[g] + y * a[g];
                }
                output[n] = Vector.Sum(acc);
            }
        }
    }
}
