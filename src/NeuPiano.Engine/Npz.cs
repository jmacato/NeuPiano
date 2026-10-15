using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace NeuPiano.Engine;

public sealed record NpyArray(string Dtype, int[] Shape, byte[] Data)
{
    public short[] ToShorts() => Dtype == "<i2"
        ? MemoryMarshal.Cast<byte, short>(Data).ToArray()
        : throw new InvalidDataException($"Expected int16 data, found {Dtype}");

    public double[] ToDoubles() => Dtype switch
    {
        "<f4" => Array.ConvertAll(MemoryMarshal.Cast<byte, float>(Data).ToArray(), v => (double)v),
        "<f8" => MemoryMarshal.Cast<byte, double>(Data).ToArray(),
        "<i2" => Array.ConvertAll(ToShorts(), v => (double)v),
        "<i4" => Array.ConvertAll(MemoryMarshal.Cast<byte, int>(Data).ToArray(), v => (double)v),
        "<i8" => Array.ConvertAll(MemoryMarshal.Cast<byte, long>(Data).ToArray(), v => (double)v),
        "|b1" => Array.ConvertAll(Data, v => (double)v),
        _ => throw new InvalidDataException($"Unsupported numeric dtype {Dtype}"),
    };

    public float[] ToFloats() => Array.ConvertAll(ToDoubles(), v => (float)v);

    public double Scalar => ToDoubles()[0];
}

public static partial class Npz
{
    public static Dictionary<string, NpyArray> Load(string path)
    {
        using var archive = ZipFile.OpenRead(path);
        var arrays = new Dictionary<string, NpyArray>();
        foreach (var entry in archive.Entries.Where(e => e.Name.EndsWith(".npy", StringComparison.Ordinal)))
        {
            using var stream = entry.Open();
            using var memory = new MemoryStream();
            stream.CopyTo(memory);
            var array = Parse(memory.ToArray());
            if (array is not null)
                arrays[entry.FullName[..^4]] = array;
        }
        return arrays;
    }

    static NpyArray? Parse(byte[] bytes)
    {
        if (bytes.Length < 10 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
            throw new InvalidDataException("Not a .npy array");
        var major = bytes[6];
        var headerLength = major == 1 ? BitConverter.ToUInt16(bytes, 8) : (int)BitConverter.ToUInt32(bytes, 8);
        var offset = major == 1 ? 10 : 12;
        var header = Encoding.ASCII.GetString(bytes, offset, headerLength);
        var dtype = DescrPattern().Match(header).Groups[1].Value;
        if (dtype.StartsWith("<U", StringComparison.Ordinal))
            return null;
        if (header.Contains("'fortran_order': True"))
            throw new InvalidDataException("Fortran-ordered arrays are not supported");
        var shape = ShapePattern().Match(header).Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();
        return new NpyArray(dtype, shape, bytes[(offset + headerLength)..]);
    }

    [GeneratedRegex(@"'descr':\s*'([^']+)'")]
    private static partial Regex DescrPattern();

    [GeneratedRegex(@"'shape':\s*\(([^)]*)\)")]
    private static partial Regex ShapePattern();
}
