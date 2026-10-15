using System;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;

namespace NeuPiano.App;

[SupportedOSPlatform("macos")]
sealed class DefaultOutputWatcher : IDisposable
{
    const string CoreAudio = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio";
    const uint SystemObject = 1, DefaultOutputDevice = 0x644F7574, GlobalScope = 0x676C6F62, MainElement = 0;
    const int SettleMs = 250;

    [StructLayout(LayoutKind.Sequential)]
    struct PropertyAddress
    {
        public uint Selector, Scope, Element;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int Listener(uint objectId, uint count, IntPtr addresses, IntPtr data);

    [DllImport(CoreAudio)]
    static extern int AudioObjectAddPropertyListener(uint objectId, ref PropertyAddress address, Listener listener, IntPtr data);

    [DllImport(CoreAudio)]
    static extern int AudioObjectRemovePropertyListener(uint objectId, ref PropertyAddress address, Listener listener, IntPtr data);

    [DllImport(CoreAudio)]
    static extern int AudioObjectGetPropertyData(uint objectId, ref PropertyAddress address, uint qualifierSize, IntPtr qualifier, ref uint size, out uint data);

    static PropertyAddress Address => new() { Selector = DefaultOutputDevice, Scope = GlobalScope, Element = MainElement };

    readonly Listener listener;
    readonly Timer settle;
    PropertyAddress address = Address;

    public DefaultOutputWatcher(Action changed)
    {
        settle = new Timer(_ => changed());
        listener = (_, _, _, _) =>
        {
            settle.Change(SettleMs, Timeout.Infinite);
            return 0;
        };
        AudioObjectAddPropertyListener(SystemObject, ref address, listener, IntPtr.Zero);
    }

    public static uint Current()
    {
        var address = Address;
        var size = (uint)sizeof(uint);
        return AudioObjectGetPropertyData(SystemObject, ref address, 0, IntPtr.Zero, ref size, out var device) == 0 ? device : 0;
    }

    public void Dispose()
    {
        AudioObjectRemovePropertyListener(SystemObject, ref address, listener, IntPtr.Zero);
        settle.Dispose();
    }
}
