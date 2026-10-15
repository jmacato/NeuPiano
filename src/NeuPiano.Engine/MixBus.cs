namespace NeuPiano.Engine;

public sealed class MixBus
{
    public const float Ceiling = .89f;
    public const int Lookahead = 96;
    static readonly float Recovery = MathF.Exp(-1f / (.2f * ModalBank.Rate));
    readonly float[] required = new float[Lookahead + 1], smoothed = new float[Lookahead], delayed = new float[2 * Lookahead];
    int requiredIndex, slot;
    float released = 1;
    double sum = Lookahead;

    public MixBus()
    {
        Array.Fill(required, 1f);
        Array.Fill(smoothed, 1f);
    }

    public void Process(float[] stereo, int frames, float volume)
    {
        for (var n = 0; n < frames; n++)
        {
            float left = stereo[2 * n] * volume, right = stereo[2 * n + 1] * volume;
            var peak = MathF.Max(MathF.Abs(left), MathF.Abs(right));
            required[requiredIndex] = peak > Ceiling ? Ceiling / peak : 1;
            requiredIndex = (requiredIndex + 1) % required.Length;
            released = MathF.Min(required.Min(), 1 - (1 - released) * Recovery);
            sum += released - smoothed[slot];
            smoothed[slot] = released;
            var gain = (float)(sum / Lookahead);
            stereo[2 * n] = delayed[2 * slot] * gain;
            stereo[2 * n + 1] = delayed[2 * slot + 1] * gain;
            delayed[2 * slot] = left;
            delayed[2 * slot + 1] = right;
            slot = (slot + 1) % Lookahead;
        }
    }
}
