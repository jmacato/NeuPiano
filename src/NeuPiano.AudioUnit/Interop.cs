using System.Runtime.InteropServices;

namespace NeuPiano.AudioUnit;

static class Status
{
    public const int Ok = 0, ParamError = -50, InvalidProperty = -10879, InvalidParameter = -10878, InvalidElement = -10877,
        FailedInitialization = -10875, TooManyFrames = -10874, FormatNotSupported = -10868, Uninitialized = -10867, InvalidScope = -10866,
        NotWritable = -10865, CannotDoNow = -10863, InvalidValue = -10851, Initialized = -10849, InvalidParameterValue = -66743;
}

static class Property
{
    public const uint ClassInfo = 0, SampleRate = 2, ParameterList = 3, ParameterInfo = 4, StreamFormat = 8, ElementCount = 11, Latency = 12,
        SupportedNumChannels = 13, MaximumFramesPerSlice = 14, ParameterValueStrings = 16, TailTime = 20, LastRenderError = 22, ContextName = 25,
        RenderQuality = 26, HostCallbacks = 27, PresentPreset = 36, OfflineRender = 37, ShouldAllocateBuffer = 51, NickName = 54, InstrumentCount = 1000;
}

[StructLayout(LayoutKind.Sequential)]
struct StreamFormat
{
    public double SampleRate;
    public uint FormatId, FormatFlags, BytesPerPacket, FramesPerPacket, BytesPerFrame, ChannelsPerFrame, BitsPerChannel, Reserved;
}

[StructLayout(LayoutKind.Sequential)]
unsafe struct AudioBuffer
{
    public uint Channels, Bytes;
    public float* Data;
}

[StructLayout(LayoutKind.Sequential)]
struct AudioBufferList
{
    public uint Count;
    public AudioBuffer First;
}

[StructLayout(LayoutKind.Sequential)]
unsafe struct ParameterInfo
{
    public fixed byte Name[52];
    public nint UnitName;
    public uint Clump;
    public nint CFName;
    public uint Unit;
    public float Min, Max, Default;
    public uint Flags;
}

[StructLayout(LayoutKind.Explicit, Size = 32)]
struct ParameterEvent
{
    [FieldOffset(0)] public uint Scope;
    [FieldOffset(4)] public uint Element;
    [FieldOffset(8)] public uint Parameter;
    [FieldOffset(12)] public uint Kind;
    [FieldOffset(20)] public float Value;
    [FieldOffset(28)] public float End;
}

[StructLayout(LayoutKind.Sequential)]
struct Preset
{
    public int Number;
    public nint Name;
}

[StructLayout(LayoutKind.Sequential)]
struct UnitParameter
{
    public nint Unit;
    public uint Id, Scope, Element;
}

[StructLayout(LayoutKind.Sequential)]
unsafe struct LibraryInfo
{
    public byte* FileName;
    public void* FileBase;
    public byte* SymbolName;
    public void* SymbolAddress;
}

static unsafe class Native
{
    [DllImport("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")]
    public static extern int AUParameterListenerNotify(nint listener, nint sender, UnitParameter* parameter);

    [DllImport("/usr/lib/libSystem.B.dylib")]
    public static extern int dladdr(void* address, LibraryInfo* info);
}
