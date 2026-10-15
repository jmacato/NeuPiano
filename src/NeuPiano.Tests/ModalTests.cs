using NeuPiano.Engine;

namespace NeuPiano.Tests;

public class ModalTests
{
    static readonly string Path = System.IO.Path.Combine(AppContext.BaseDirectory, "reference_modal.npz");
    static readonly Dictionary<string, NpyArray> Reference = Npz.Load(Path);
    static readonly ModalBank Bank = ModalBank.Load(Path);
    static readonly ControlNet Net = ControlNet.Load(Path);

    static float[] Render(ModalPiano piano, int frames, int block = 500)
    {
        var stereo = new float[2 * frames];
        for (var start = 0; start < frames; start += block)
            piano.Render(stereo, start, Math.Min(block, frames - start));
        return stereo;
    }

    static float Peak(float[] stereo, int channel, int from, int to)
    {
        var peak = 0f;
        for (var n = from; n < to; n++)
            peak = MathF.Max(peak, MathF.Abs(stereo[2 * n + channel]));
        return peak;
    }

    [Theory]
    [InlineData(false, "tonal")]
    [InlineData(true, "controlled")]
    public void StringPartialsMatchThePythonRender(bool controlled, string prefix)
    {
        var piano = new ModalPiano(Bank, controlled ? Net : null) { NoiseOn = false };
        piano.NoteOn(60, 127);
        var stereo = Render(piano, 48000, 333);
        foreach (var (channel, name) in new[] { (0, $"{prefix}_left"), (1, $"{prefix}_right") })
        {
            var expected = Reference[name].ToFloats();
            var peak = expected.Max(MathF.Abs);
            var error = 0f;
            for (var n = 0; n < expected.Length; n++)
                error = MathF.Max(error, MathF.Abs(stereo[2 * n + channel] - expected[n]));
            Assert.True(error < 1e-3f * peak, $"{name}: max error {error / peak:E2} of peak");
        }
    }

    [Theory]
    [InlineData(false, "noise_rms")]
    [InlineData(true, "controlled_noise_rms")]
    public void NoiseLevelMatchesThePythonRender(bool controlled, string name)
    {
        var piano = new ModalPiano(Bank, controlled ? Net : null) { PartialsOn = false };
        piano.NoteOn(60, 127);
        var stereo = Render(piano, 96000);
        var expected = Reference[name].ToFloats();
        var differences = new double[expected.Length];
        for (var w = 0; w < expected.Length; w++)
        {
            var energy = 0.0;
            for (var n = w * 4800; n < (w + 1) * 4800; n++)
                energy += stereo[2 * n] * (double)stereo[2 * n];
            differences[w] = Math.Abs(10 * Math.Log10(energy / 4800 / (expected[w] * (double)expected[w])));
        }
        Array.Sort(differences);
        Assert.True(differences[differences.Length / 2] < 1, $"median level difference {differences[differences.Length / 2]:0.00} dB");
    }

    [Fact]
    public void ControlNetMatchesThePythonForwardPass()
    {
        var expected = Reference["controls"].ToFloats();
        var result = new float[Net.Outputs];
        var error = 0f;
        for (var frame = 0; frame < 701; frame += 7)
        {
            Net.Evaluate(60, 9, frame, result);
            for (var o = 0; o < result.Length; o++)
                error = MathF.Max(error, MathF.Abs(result[o] - expected[frame * Net.Outputs + o]));
        }
        Assert.True(error < 1e-4f, $"max log-gain error {error:E2}");
    }

    [Fact]
    public void VelocityScalesTheLevelOverThirtyDecibels()
    {
        Assert.Equal(1f, ModalPiano.VelocityGain(127));
        Assert.Equal(-30 * 126 / 127f, 20 * MathF.Log10(ModalPiano.VelocityGain(1)), 3);
        var levels = new List<float>();
        foreach (var velocity in new[] { 104, 93 })
        {
            var piano = new ModalPiano(Bank);
            piano.NoteOn(60, velocity);
            levels.Add(Peak(Render(piano, 4800), 0, 0, 4800));
        }
        Assert.Equal(Bank.Layer(104), Bank.Layer(93));
        Assert.Equal(30 * 11 / 127f, 20 * MathF.Log10(levels[0] / levels[1]), 2);
    }

    [Fact]
    public void VelocityCurveDefaultsToIdentityAndHardnessRaisesSoftNotes()
    {
        Assert.Equal(Enumerable.Range(0, 128).Select(v => (byte)v), new PianoSettings().VelocityTable());
        var hard = new PianoSettings { VelocityHardness = .5f }.VelocityTable();
        var soft = new PianoSettings { VelocityHardness = -.5f }.VelocityTable();
        Assert.True(hard[64] > 64 && soft[64] < 64);
        Assert.Equal(127, hard[127]);
        var range = new PianoSettings { VelocityMin = 20, VelocityMax = 100 }.VelocityTable();
        Assert.Equal((0, 127, 127), (range[20], range[100], range[110]));
    }

    [Fact]
    public void DampersAndSustainPedal()
    {
        var piano = new ModalPiano(Bank);
        piano.NoteOn(60, 100);
        Render(piano, 4800);
        piano.NoteOff(60);
        var damped = Render(piano, 48000);
        Render(piano, 48000);
        Assert.Equal(0, piano.ActiveVoices);

        piano.Controller(64, 127);
        piano.NoteOn(60, 100);
        Render(piano, 4800);
        piano.NoteOff(60);
        var held = Render(piano, 48000);
        Assert.Equal(1, piano.ActiveVoices);
        Assert.True(Peak(held, 0, 40000, 48000) > 30 * Peak(damped, 0, 40000, 48000));
        piano.Controller(64, 0);
        Render(piano, 96000);
        Assert.Equal(0, piano.ActiveVoices);
    }

    [Fact]
    public void VelocityPicksTheRecordedLayerAndUnknownKeysAreIgnored()
    {
        Assert.Equal(0, Bank.Layer(1));
        Assert.Equal(Bank.Layers - 1, Bank.Layer(127));
        var piano = new ModalPiano(Bank);
        piano.NoteOn(20, 100);
        piano.NoteOn(61, 100);
        Assert.Equal(0, piano.ActiveVoices);
        piano.NoteOn(60, 30);
        var soft = Peak(Render(piano, 4800), 0, 0, 4800);
        piano.Controller(120, 0);
        piano.NoteOn(60, 127);
        var loud = Peak(Render(piano, 4800), 0, 0, 4800);
        Assert.True(loud > 2 * soft);
    }
}
