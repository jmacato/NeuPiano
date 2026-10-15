using NeuPiano.Engine;

namespace NeuPiano.Tests;

public class ReleaseTests
{
    static readonly ModalBank Bank = ModalBank.Load(Path.Combine(AppContext.BaseDirectory, "reference_modal.npz"));
    static readonly ModalBank Releases = ModalBank.Load(Path.Combine(AppContext.BaseDirectory, "reference_release.npz"));

    static float[] Render(ModalPiano piano, int frames)
    {
        var stereo = new float[2 * frames];
        for (var start = 0; start < frames; start += 500)
            piano.Render(stereo, start, Math.Min(500, frames - start));
        return stereo;
    }

    static double Rms(float[] stereo, int from, int to)
    {
        var sum = 0.0;
        for (var n = from; n < to; n++)
            sum += stereo[2 * n] * (double)stereo[2 * n];
        return Math.Sqrt(sum / (to - from));
    }

    [Theory]
    [InlineData(300, 50, 4)]
    [InlineData(300, 100, 5)]
    [InlineData(300, 120, 6)]
    [InlineData(1000, 50, 4)]
    [InlineData(1000, 100, 1)]
    [InlineData(3000, 100, 2)]
    [InlineData(10000, 100, 3)]
    public void HoldTimeAndVelocityPickTheReleaseLayer(double milliseconds, int velocity, int layer) =>
        Assert.Equal(layer, ModalPiano.ReleaseLayer(Releases.Thresholds, milliseconds, velocity));

    [Fact]
    public void DampersFallAtOneHundredDecibelsPerSecond()
    {
        var piano = new ModalPiano(Bank) { NoiseOn = false };
        piano.NoteOn(60, 100);
        Render(piano, 24000);
        piano.NoteOff(60);
        var tail = Render(piano, 24000);
        var drop = 20 * Math.Log10(Rms(tail, 2400, 4800) / Rms(tail, 12000, 14400));
        Assert.InRange(drop, 19.0, 26.0);
    }

    [Fact]
    public void TransposeAndReleaseTimeReachTheModalSynth()
    {
        var piano = new ModalPiano(Bank) { NoiseOn = false };
        piano.Apply(new PianoSettings { Transpose = -1, ReleaseTime = 2 });
        piano.NoteOn(60, 100);
        Assert.Equal(0, piano.ActiveVoices);
        piano.NoteOn(61, 100);
        Assert.Equal(1, piano.ActiveVoices);
        Render(piano, 24000);
        piano.NoteOff(61);
        var tail = Render(piano, 24000);
        var drop = 20 * Math.Log10(Rms(tail, 2400, 4800) / Rms(tail, 12000, 14400));
        Assert.InRange(drop, 9.5, 14.0);
    }

    [Fact]
    public void KeyUpAddsTheReleaseSoundOnlyOnce()
    {
        double After(bool releases, out int voices)
        {
            var piano = new ModalPiano(Bank, null, Releases) { ReleasesOn = releases };
            piano.NoteOn(60, 100);
            Render(piano, 48000);
            piano.NoteOff(60);
            voices = piano.ActiveVoices;
            piano.Controller(64, 127);
            piano.Controller(64, 0);
            voices = Math.Max(voices, piano.ActiveVoices);
            return Rms(Render(piano, 48000), 24000, 48000);
        }
        var plain = After(false, out var plainVoices);
        var released = After(true, out var releasedVoices);
        Assert.Equal(1, plainVoices);
        Assert.Equal(2, releasedVoices);
        Assert.True(released > 2 * plain);
    }
}
