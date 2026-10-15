using System.Runtime.InteropServices;

namespace NeuPiano.AudioUnit;

[StructLayout(LayoutKind.Sequential)]
unsafe struct PlugIn
{
    public delegate* unmanaged<PlugIn*, nint, int> Open;
    public delegate* unmanaged<PlugIn*, int> Close;
    public delegate* unmanaged<short, void*> Lookup;
    public nint Reserved;
    public nint Instrument;
}

static unsafe class Exports
{
    [UnmanagedCallersOnly(EntryPoint = "NeuPianoFactory")]
    public static PlugIn* Factory(void* description)
    {
        var plugIn = (PlugIn*)NativeMemory.AllocZeroed((nuint)sizeof(PlugIn));
        plugIn->Open = &Open;
        plugIn->Close = &Close;
        plugIn->Lookup = &Lookup;
        return plugIn;
    }

    static Instrument Of(PlugIn* self) => (Instrument)GCHandle.FromIntPtr(self->Instrument).Target!;

    [UnmanagedCallersOnly]
    static int Open(PlugIn* self, nint instance)
    {
        try
        {
            self->Instrument = GCHandle.ToIntPtr(GCHandle.Alloc(new Instrument(instance)));
            return Status.Ok;
        }
        catch (Exception)
        {
            return Status.FailedInitialization;
        }
    }

    [UnmanagedCallersOnly]
    static int Close(PlugIn* self)
    {
        try
        {
            if (self->Instrument != 0)
            {
                var handle = GCHandle.FromIntPtr(self->Instrument);
                (handle.Target as Instrument)?.Dispose();
                handle.Free();
            }
        }
        catch (Exception)
        {
        }
        NativeMemory.Free(self);
        return Status.Ok;
    }

    [UnmanagedCallersOnly]
    static void* Lookup(short selector) => selector switch
    {
        0x0001 => (delegate* unmanaged<PlugIn*, int>)&Initialize,
        0x0002 => (delegate* unmanaged<PlugIn*, int>)&Uninitialize,
        0x0003 => (delegate* unmanaged<PlugIn*, uint, uint, uint, uint*, byte*, int>)&GetPropertyInfo,
        0x0004 => (delegate* unmanaged<PlugIn*, uint, uint, uint, void*, uint*, int>)&GetProperty,
        0x0005 => (delegate* unmanaged<PlugIn*, uint, uint, uint, void*, uint, int>)&SetProperty,
        0x0006 => (delegate* unmanaged<PlugIn*, uint, uint, uint, float*, int>)&GetParameter,
        0x0007 => (delegate* unmanaged<PlugIn*, uint, uint, uint, float, uint, int>)&SetParameter,
        0x0009 => (delegate* unmanaged<PlugIn*, uint, uint, int>)&Reset,
        0x000A => (delegate* unmanaged<PlugIn*, uint, nint, nint, int>)&AddPropertyListener,
        0x000B => (delegate* unmanaged<PlugIn*, uint, nint, int>)&RemovePropertyListener,
        0x000E => (delegate* unmanaged<PlugIn*, uint*, void*, uint, uint, AudioBufferList*, int>)&Render,
        0x000F => (delegate* unmanaged<PlugIn*, nint, nint, int>)&AddRenderNotify,
        0x0010 => (delegate* unmanaged<PlugIn*, nint, nint, int>)&RemoveRenderNotify,
        0x0011 => (delegate* unmanaged<PlugIn*, ParameterEvent*, uint, int>)&ScheduleParameters,
        0x0012 => (delegate* unmanaged<PlugIn*, uint, nint, nint, int>)&RemovePropertyListenerWithUserData,
        0x0101 => (delegate* unmanaged<PlugIn*, uint, uint, uint, uint, int>)&MidiEvent,
        0x0102 => (delegate* unmanaged<PlugIn*, byte*, uint, int>)&SysEx,
        _ => null,
    };

    [UnmanagedCallersOnly]
    static int Initialize(PlugIn* self)
    {
        try
        {
            return Of(self).Initialize();
        }
        catch (Exception)
        {
            return Status.FailedInitialization;
        }
    }

    [UnmanagedCallersOnly]
    static int Uninitialize(PlugIn* self)
    {
        try
        {
            return Of(self).Uninitialize();
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int GetPropertyInfo(PlugIn* self, uint id, uint scope, uint element, uint* size, byte* writable)
    {
        try
        {
            var status = Of(self).Info(id, scope, element, out var bytes, out var canWrite);
            if (status == Status.Ok)
            {
                if (size != null)
                    *size = bytes;
                if (writable != null)
                    *writable = canWrite ? (byte)1 : (byte)0;
            }
            return status;
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int GetProperty(PlugIn* self, uint id, uint scope, uint element, void* data, uint* size)
    {
        try
        {
            return Of(self).GetProperty(id, scope, element, data, size);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int SetProperty(PlugIn* self, uint id, uint scope, uint element, void* data, uint size)
    {
        try
        {
            return Of(self).SetProperty(id, scope, element, data, size);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int GetParameter(PlugIn* self, uint id, uint scope, uint element, float* value)
    {
        try
        {
            return Of(self).GetParameter(id, scope, value);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int SetParameter(PlugIn* self, uint id, uint scope, uint element, float value, uint offset)
    {
        try
        {
            return Of(self).SetParameter(id, scope, value);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int ScheduleParameters(PlugIn* self, ParameterEvent* events, uint count)
    {
        try
        {
            return Of(self).ScheduleParameters(events, count);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int Reset(PlugIn* self, uint scope, uint element)
    {
        try
        {
            return Of(self).Reset();
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int AddPropertyListener(PlugIn* self, uint id, nint proc, nint data)
    {
        try
        {
            return Of(self).AddListener(id, proc, data);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int RemovePropertyListener(PlugIn* self, uint id, nint proc)
    {
        try
        {
            return Of(self).RemoveListener(id, proc, 0, any: true);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int RemovePropertyListenerWithUserData(PlugIn* self, uint id, nint proc, nint data)
    {
        try
        {
            return Of(self).RemoveListener(id, proc, data, any: false);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int AddRenderNotify(PlugIn* self, nint proc, nint data)
    {
        try
        {
            return Of(self).AddNotify(proc, data);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int RemoveRenderNotify(PlugIn* self, nint proc, nint data)
    {
        try
        {
            return Of(self).RemoveNotify(proc, data);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int Render(PlugIn* self, uint* flags, void* time, uint bus, uint frames, AudioBufferList* io)
    {
        try
        {
            return Of(self).Render(flags, time, bus, frames, io);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int MidiEvent(PlugIn* self, uint status, uint data1, uint data2, uint offset)
    {
        try
        {
            return Of(self).Midi(status, data1, data2, offset);
        }
        catch (Exception)
        {
            return Status.CannotDoNow;
        }
    }

    [UnmanagedCallersOnly]
    static int SysEx(PlugIn* self, byte* data, uint length) => Status.Ok;
}
