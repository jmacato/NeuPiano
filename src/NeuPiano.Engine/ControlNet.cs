using System.Numerics;
using System.Runtime.InteropServices;

namespace NeuPiano.Engine;

public sealed class ControlNet
{
    public const int Frame = 960, Kinds = 2;
    const int Rows = 8;

    readonly float[] embedding;
    readonly float[][] weights, biases;
    readonly int[] inputs, outputs, strides;
    readonly float[] a, b;
    readonly int harmonics, dimensions, layers;
    readonly double origin, span, onset;
    readonly float bound;

    ControlNet(Dictionary<string, NpyArray> arrays)
    {
        var scalars = arrays["scalars"].ToDoubles();
        harmonics = (int)scalars[0];
        (origin, span, bound) = (scalars[1], scalars[2], (float)scalars[3]);
        onset = scalars.Length > 7 ? scalars[7] : 0;
        var depth = (int)scalars[4];
        FirstNote = (int)scalars[5];
        Keys = (int)scalars[6];
        Lows = arrays["lows"].ToFloats();
        embedding = arrays["embedding"].ToFloats();
        dimensions = arrays["embedding"].Shape[2];
        layers = arrays["embedding"].Shape[1];
        (weights, biases, inputs, outputs, strides) = (new float[depth][], new float[depth][], new int[depth], new int[depth], new int[depth]);
        var lanes = Vector<float>.Count;
        for (var i = 0; i < depth; i++)
        {
            var weight = arrays[$"weight{i}"];
            (outputs[i], inputs[i]) = (weight.Shape[0], weight.Shape[1]);
            strides[i] = (inputs[i] + lanes - 1) / lanes * lanes;
            var values = weight.ToFloats();
            weights[i] = new float[(outputs[i] + Rows - 1) / Rows * Rows * strides[i]];
            for (var o = 0; o < outputs[i]; o++)
                values.AsSpan(o * inputs[i], inputs[i]).CopyTo(weights[i].AsSpan(o * strides[i]));
            biases[i] = arrays[$"bias{i}"].ToFloats();
            if (biases[i].Length != outputs[i] || (i > 0 && inputs[i] != outputs[i - 1]))
                throw new InvalidDataException($"Control net layer {i} does not match its neighbours");
        }
        if (inputs[0] != dimensions + 3 + 2 * harmonics || embedding.Length != Keys * layers * dimensions ||
            Outputs != ModalBank.Channels * Kinds * Bands)
            throw new InvalidDataException("Control net arrays do not match its key, layer and band counts");
        a = new float[(Math.Max(strides.Max(), outputs.Max()) + Rows - 1) / Rows * Rows];
        b = new float[a.Length];
    }

    public static ControlNet Load(string path) => new(Npz.Load(path));

    public int FirstNote { get; }
    public int Keys { get; }
    public int Bands => Lows.Length;
    public int Outputs => outputs[^1];
    internal float[] Lows { get; }

    public bool Contains(int note) => note >= FirstNote && note < FirstNote + Keys;

    public int Band(double frequency)
    {
        var band = 0;
        while (band + 1 < Lows.Length && Lows[band + 1] <= frequency)
            band++;
        return band;
    }

    public bool Changes(int frame) => frame * (double)Frame / ModalBank.Rate <= span;

    public void Evaluate(int note, int layer, int frame, Span<float> result)
    {
        var key = note - FirstNote;
        embedding.AsSpan((key * layers + layer) * dimensions, dimensions).CopyTo(a);
        var n = dimensions;
        a[n++] = Keys > 1 ? key / (float)(Keys - 1) : 0;
        a[n++] = layer / (float)(layers - 1);
        var t = Math.Clamp(frame * (double)Frame / ModalBank.Rate, 0, span);
        var u = (float)(Math.Log(1 + t / origin) / Math.Log(1 + span / origin));
        a[n++] = u;
        for (var k = 1; k <= harmonics; k++)
        {
            a[n + k - 1] = MathF.Sin(MathF.PI * k * u);
            a[n + harmonics + k - 1] = MathF.Cos(MathF.PI * k * u);
        }
        a.AsSpan(n + 2 * harmonics).Clear();
        var (x, y) = (a, b);
        for (var l = 0; l < weights.Length; l++)
        {
            Multiply(weights[l], strides[l], outputs[l], x, y);
            for (var o = 0; o < outputs[l]; o++)
            {
                var v = biases[l][o] + y[o];
                y[o] = l < weights.Length - 1 ? v / (1 + MathF.Exp(-v)) : bound * MathF.Tanh(v / bound);
            }
            y.AsSpan(outputs[l]).Clear();
            (x, y) = (y, x);
        }
        var weight = onset > 0 ? (float)Math.Min(t / onset, 1) : 1;
        for (var o = 0; o < Outputs; o++)
            result[o] = x[o] * weight;
    }

    static void Multiply(float[] weight, int stride, int count, float[] input, float[] output)
    {
        var x = MemoryMarshal.Cast<float, Vector<float>>(input.AsSpan(0, stride));
        var w = MemoryMarshal.Cast<float, Vector<float>>(weight.AsSpan());
        var step = stride / Vector<float>.Count;
        for (var o = 0; o < count; o += Rows)
        {
            var r = o * step;
            Vector<float> s0 = default, s1 = default, s2 = default, s3 = default, s4 = default, s5 = default, s6 = default, s7 = default;
            for (var i = 0; i < x.Length; i++, r++)
            {
                var v = x[i];
                s0 = Vector.FusedMultiplyAdd(w[r], v, s0);
                s1 = Vector.FusedMultiplyAdd(w[r + step], v, s1);
                s2 = Vector.FusedMultiplyAdd(w[r + 2 * step], v, s2);
                s3 = Vector.FusedMultiplyAdd(w[r + 3 * step], v, s3);
                s4 = Vector.FusedMultiplyAdd(w[r + 4 * step], v, s4);
                s5 = Vector.FusedMultiplyAdd(w[r + 5 * step], v, s5);
                s6 = Vector.FusedMultiplyAdd(w[r + 6 * step], v, s6);
                s7 = Vector.FusedMultiplyAdd(w[r + 7 * step], v, s7);
            }
            (output[o], output[o + 1], output[o + 2], output[o + 3]) = (Vector.Sum(s0), Vector.Sum(s1), Vector.Sum(s2), Vector.Sum(s3));
            (output[o + 4], output[o + 5], output[o + 6], output[o + 7]) = (Vector.Sum(s4), Vector.Sum(s5), Vector.Sum(s6), Vector.Sum(s7));
        }
    }
}
