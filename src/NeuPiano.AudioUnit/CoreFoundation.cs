using System.Runtime.InteropServices;
using System.Text;

namespace NeuPiano.AudioUnit;

static unsafe class CF
{
    const string Library = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation";
    const nint SInt32 = 3;
    const uint Utf8 = 0x08000100;
    static readonly nint library = NativeLibrary.Load(Library);
    static readonly nint keyCallbacks = NativeLibrary.GetExport(library, "kCFTypeDictionaryKeyCallBacks");
    static readonly nint valueCallbacks = NativeLibrary.GetExport(library, "kCFTypeDictionaryValueCallBacks");
    static readonly nint arrayCallbacks = NativeLibrary.GetExport(library, "kCFTypeArrayCallBacks");
    static readonly nuint dictionaryType = CFDictionaryGetTypeID(), dataType = CFDataGetTypeID(), numberType = CFNumberGetTypeID(), stringType = CFStringGetTypeID();

    [DllImport(Library)] public static extern void CFRelease(nint value);
    [DllImport(Library)] public static extern nint CFRetain(nint value);
    [DllImport(Library)] static extern nint CFDictionaryCreateMutable(nint allocator, nint capacity, nint keys, nint values);
    [DllImport(Library)] static extern void CFDictionarySetValue(nint dictionary, nint key, nint value);
    [DllImport(Library)] static extern nint CFDictionaryGetValue(nint dictionary, nint key);
    [DllImport(Library)] static extern nint CFNumberCreate(nint allocator, nint type, void* value);
    [DllImport(Library)] static extern byte CFNumberGetValue(nint number, nint type, void* value);
    [DllImport(Library)] static extern nint CFStringCreateWithBytes(nint allocator, byte* bytes, nint length, uint encoding, byte external);
    [DllImport(Library)] static extern nint CFStringGetLength(nint value);
    [DllImport(Library)] static extern byte CFStringGetCString(nint value, byte* buffer, nint size, uint encoding);
    [DllImport(Library)] static extern nint CFDataCreate(nint allocator, byte* bytes, nint length);
    [DllImport(Library)] static extern nint CFDataGetLength(nint data);
    [DllImport(Library)] static extern byte* CFDataGetBytePtr(nint data);
    [DllImport(Library)] static extern nint CFArrayCreate(nint allocator, nint* values, nint count, nint callbacks);
    [DllImport(Library)] static extern nuint CFGetTypeID(nint value);
    [DllImport(Library)] static extern nuint CFDictionaryGetTypeID();
    [DllImport(Library)] static extern nuint CFDataGetTypeID();
    [DllImport(Library)] static extern nuint CFNumberGetTypeID();
    [DllImport(Library)] static extern nuint CFStringGetTypeID();

    public static nint String(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        fixed (byte* p = bytes)
            return CFStringCreateWithBytes(0, p, bytes.Length, Utf8, 0);
    }

    public static string? ToString(nint value)
    {
        if (value == 0 || CFGetTypeID(value) != stringType)
            return null;
        var size = (int)CFStringGetLength(value) * 4 + 1;
        var buffer = new byte[size];
        fixed (byte* p = buffer)
            return CFStringGetCString(value, p, size, Utf8) != 0 ? Marshal.PtrToStringUTF8((nint)p) : null;
    }

    public static bool IsString(nint value) => value != 0 && CFGetTypeID(value) == stringType;

    public static nint Dictionary() => CFDictionaryCreateMutable(0, 0, keyCallbacks, valueCallbacks);

    public static bool IsDictionary(nint value) => value != 0 && CFGetTypeID(value) == dictionaryType;

    public static void Set(nint dictionary, string key, nint value)
    {
        var name = String(key);
        CFDictionarySetValue(dictionary, name, value);
        CFRelease(name);
        CFRelease(value);
    }

    public static nint Get(nint dictionary, string key)
    {
        var name = String(key);
        var value = CFDictionaryGetValue(dictionary, name);
        CFRelease(name);
        return value;
    }

    public static nint Number(int value) => CFNumberCreate(0, SInt32, &value);

    public static int? Int(nint value)
    {
        if (value == 0 || CFGetTypeID(value) != numberType)
            return null;
        int result;
        CFNumberGetValue(value, SInt32, &result);
        return result;
    }

    public static nint Data(ReadOnlySpan<byte> bytes)
    {
        fixed (byte* p = bytes)
            return CFDataCreate(0, p, bytes.Length);
    }

    public static ReadOnlySpan<byte> Bytes(nint value) =>
        value != 0 && CFGetTypeID(value) == dataType ? new ReadOnlySpan<byte>(CFDataGetBytePtr(value), (int)CFDataGetLength(value)) : default;

    public static nint Array(string[] values)
    {
        var strings = values.Select(String).ToArray();
        nint result;
        fixed (nint* p = strings)
            result = CFArrayCreate(0, p, strings.Length, arrayCallbacks);
        foreach (var s in strings)
            CFRelease(s);
        return result;
    }
}
