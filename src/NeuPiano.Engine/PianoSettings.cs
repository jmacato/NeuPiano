namespace NeuPiano.Engine;

public sealed class PianoSettings
{
    public float DynamicRange { get; set; } = ModalPiano.DynamicRangeDb;
    public float ReleaseTime { get; set; } = 1;
    public bool Releases { get; set; } = true;
    public int Voices { get; set; } = ModalPiano.MaxVoices;
    public int Octave { get; set; }
    public int Transpose { get; set; }
    public int Tune { get; set; }
    public float A440 { get; set; } = 440;
    public float VelocityHardness { get; set; }
    public int VelocityMin { get; set; }
    public int VelocityMax { get; set; } = 127;

    public PianoSettings Clone() => (PianoSettings)MemberwiseClone();

    public byte[] VelocityTable()
    {
        var table = new byte[128];
        var exponent = Math.Pow(4, -Math.Clamp(VelocityHardness, -1, 1));
        for (var v = 0; v < table.Length; v++)
        {
            var x = VelocityMax > VelocityMin ? Math.Clamp((v - VelocityMin) / (double)(VelocityMax - VelocityMin), 0, 1) : v >= VelocityMax ? 1 : 0;
            table[v] = (byte)Math.Round(127 * Math.Pow(x, exponent));
        }
        return table;
    }
}
