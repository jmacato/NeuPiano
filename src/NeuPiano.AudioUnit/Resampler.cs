using System.Numerics;
using System.Runtime.InteropServices;
using NeuPiano.Engine;

namespace NeuPiano.AudioUnit;

sealed unsafe class Resampler
{
    const int Phases = 256;
    const double Beta = 8, Passband = .94;
    readonly float[] table, block;
    readonly float[][] fifo;
    readonly Action<float[], int> source;
    readonly int half, taps;
    int length;
    long dropped;
    double position;

    public Resampler(double rate, int maxFrames, Action<float[], int> source)
    {
        this.source = source;
        Step = ModalBank.Rate / rate;
        var ratio = Math.Min(1, rate / ModalBank.Rate);
        taps = Taps(rate);
        half = taps / 2;
        var cutoff = .5 * ratio * Passband;
        table = new float[(Phases + 1) * taps];
        for (var p = 0; p <= Phases; p++)
        {
            var row = table.AsSpan(p * taps, taps);
            var sum = 0.0;
            for (var k = 0; k < taps; k++)
            {
                var x = k - half + 1 - p / (double)Phases;
                var v = 2 * cutoff * Sinc(2 * cutoff * x) * Kaiser(x / half);
                row[k] = (float)v;
                sum += v;
            }
            for (var k = 0; k < taps; k++)
                row[k] = (float)(row[k] / sum);
        }
        var capacity = taps + (int)Math.Ceiling(maxFrames * Step) + 8;
        fifo = [new float[capacity], new float[capacity]];
        block = new float[2 * capacity];
        Reset(0);
    }

    public double Step { get; }

    public static int Taps(double rate) => ((int)Math.Ceiling(96 / Math.Min(1, rate / ModalBank.Rate)) + 7) / 8 * 8;
    public long Delay => half + 1;
    public double Time => position - (half - 1) + dropped;

    public void Reset(long written)
    {
        Array.Clear(fifo[0]);
        Array.Clear(fifo[1]);
        length = half - 1;
        position = half - 1;
        dropped = written;
    }

    public void Process(float* left, float* right, int frames)
    {
        var start = (int)position - half + 1;
        if (start > 0)
        {
            for (var c = 0; c < 2; c++)
                Array.Copy(fifo[c], start, fifo[c], 0, length - start);
            length -= start;
            position -= start;
            dropped += start;
        }
        var need = (int)(position + (frames - 1) * Step) + half + 1 - length;
        if (need > 0)
        {
            source(block, need);
            for (var n = 0; n < need; n++)
            {
                fifo[0][length + n] = block[2 * n];
                fifo[1][length + n] = block[2 * n + 1];
            }
            length += need;
        }
        for (var n = 0; n < frames; n++)
        {
            var at = (int)position;
            var p = (position - at) * Phases;
            var row = (int)p;
            var t = (float)(p - row);
            var a = MemoryMarshal.Cast<float, Vector<float>>(table.AsSpan(row * taps, taps));
            var b = MemoryMarshal.Cast<float, Vector<float>>(table.AsSpan((row + 1) * taps, taps));
            var l = MemoryMarshal.Cast<float, Vector<float>>(fifo[0].AsSpan(at - half + 1, taps));
            var r = MemoryMarshal.Cast<float, Vector<float>>(fifo[1].AsSpan(at - half + 1, taps));
            Vector<float> la = default, lb = default, ra = default, rb = default;
            for (var k = 0; k < a.Length; k++)
            {
                la = Vector.FusedMultiplyAdd(a[k], l[k], la);
                lb = Vector.FusedMultiplyAdd(b[k], l[k], lb);
                ra = Vector.FusedMultiplyAdd(a[k], r[k], ra);
                rb = Vector.FusedMultiplyAdd(b[k], r[k], rb);
            }
            float l0 = Vector.Sum(la), r0 = Vector.Sum(ra);
            left[n] = l0 + t * (Vector.Sum(lb) - l0);
            right[n] = r0 + t * (Vector.Sum(rb) - r0);
            position += Step;
        }
    }

    static double Sinc(double x) => x == 0 ? 1 : Math.Sin(Math.PI * x) / (Math.PI * x);

    static double Kaiser(double x) => Math.Abs(x) > 1 ? 0 : Bessel(Beta * Math.Sqrt(1 - x * x)) / Bessel(Beta);

    static double Bessel(double x)
    {
        double sum = 1, term = 1;
        for (var k = 1; k < 50; k++)
        {
            term *= x * x / (4.0 * k * k);
            sum += term;
        }
        return sum;
    }
}
