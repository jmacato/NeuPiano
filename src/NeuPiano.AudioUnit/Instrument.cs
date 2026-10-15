using System.Runtime.InteropServices;
using System.Text;
using NeuPiano.Engine;

namespace NeuPiano.AudioUnit;

readonly record struct Parameter(string Name, uint Unit, float Min, float Max, float Default);

record struct Event(int Offset, byte Status, byte Data1, byte Data2);

record struct Scheduled(long Target, byte Status, byte Data1, byte Data2);

sealed unsafe class Instrument : IDisposable
{
    public const uint Type = 0x61756D75, Subtype = 0x506E6F31, Manufacturer = 0x4E657550;
    const uint Lpcm = 0x6C70636D, Global = 0, Input = 1, Output = 2, Group = 3;
    const uint Readable = 1u << 30, Writable = 1u << 31, HasName = 1u << 27;
    const int Queue = 4096, Volume = 0;

    public static readonly Parameter[] Parameters =
    [
        new("Volume", 13, -60, 12, -12),
        new("Dynamic Range", 13, 0, 60, ModalPiano.DynamicRangeDb),
        new("Release Time", 0, .2f, 3, 1),
        new("Release Sounds", 2, 0, 1, 1),
        new("Control Net", 2, 0, 1, 1),
        new("Voices", 0, 4, ModalPiano.MaxVoices, ModalPiano.MaxVoices),
        new("Transpose", 10, -12, 12, 0),
        new("Fine Tune", 9, -99, 99, 0),
        new("A4", 8, 420, 460, 440),
        new("Velocity Hardness", 0, -100, 100, 0),
        new("Velocity Min", 0, 0, 127, 0),
        new("Velocity Max", 0, 0, 127, 127),
    ];

    static readonly nint[] names = Parameters.Select(p => CF.String(p.Name)).ToArray();
    static readonly Lazy<(ModalBank Bank, ModalBank? Releases, string? Controls)> banks = new(LoadBanks);

    readonly nint handle;
    readonly float[] values = Parameters.Select(p => p.Default).ToArray();
    readonly Lock gate = new();
    readonly Event[] incoming = new Event[Queue], batch = new Event[Queue];
    readonly Scheduled[] schedule = new Scheduled[Queue];
    readonly byte[] hostCallbacks = new byte[40];
    readonly PianoSettings settings = new();
    (uint Id, nint Proc, nint Data)[] listeners = [];
    (nint Proc, nint Data)[] notifies = [];
    double rate = 44100;
    uint maxFrames = 1156, quality = 127, offline, allocate = 1;
    int incomingCount, head, scheduled, version = 1, applied, lastError, reset, presetNumber = -1;
    nint presetName = CF.String("Untitled"), contextName, nickName;
    bool initialized;
    ModalPiano? piano;
    Resampler? resampler;
    float[] scratch = [];
    float* left, right;
    long written;
    float gain;

    public Instrument(nint handle) => this.handle = handle;

    static (ModalBank, ModalBank?, string?) LoadBanks()
    {
        var folder = Resources();
        var releases = Path.Combine(folder, "release_bank.npz");
        var controls = Path.Combine(folder, "control_net.npz");
        return (ModalBank.Load(Path.Combine(folder, "modal_bank.npz")), File.Exists(releases) ? ModalBank.Load(releases) : null,
                File.Exists(controls) ? controls : null);
    }

    static string Resources()
    {
        LibraryInfo info;
        delegate* unmanaged<void*, PlugIn*> factory = &Exports.Factory;
        if (Native.dladdr(factory, &info) == 0 || info.FileName == null)
            throw new DllNotFoundException("Could not locate the NeuPiano bundle");
        var binary = Marshal.PtrToStringUTF8((nint)info.FileName)!;
        return Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(binary))!, "Resources");
    }

    public int Initialize()
    {
        if (initialized)
            return Status.Ok;
        var (bank, releases, controls) = banks.Value;
        piano = new ModalPiano(bank, controls is null ? null : ControlNet.Load(controls), releases);
        Apply(force: true);
        written = head = scheduled = 0;
        lock (gate)
            incomingCount = 0;
        scratch = new float[2 * maxFrames];
        resampler = rate == ModalBank.Rate ? null : new Resampler(rate, (int)maxFrames, Synthesize);
        left = (float*)NativeMemory.AllocZeroed(maxFrames, sizeof(float));
        right = (float*)NativeMemory.AllocZeroed(maxFrames, sizeof(float));
        gain = Gain();
        initialized = true;
        return Status.Ok;
    }

    public int Uninitialize()
    {
        initialized = false;
        piano = null;
        resampler = null;
        Free();
        return Status.Ok;
    }

    void Free()
    {
        NativeMemory.Free(left);
        NativeMemory.Free(right);
        left = right = null;
    }

    public void Dispose()
    {
        Free();
        foreach (var name in (ReadOnlySpan<nint>)[presetName, contextName, nickName])
            if (name != 0)
                CF.CFRelease(name);
        presetName = contextName = nickName = 0;
    }

    float Gain() => MathF.Pow(10, values[Volume] / 20);

    void Apply(bool force = false)
    {
        var current = Volatile.Read(ref version);
        if (!force && current == applied)
            return;
        applied = current;
        settings.DynamicRange = values[1];
        settings.ReleaseTime = values[2];
        settings.Releases = values[3] >= .5f;
        settings.Voices = (int)MathF.Round(values[5]);
        settings.Transpose = (int)MathF.Round(values[6]);
        settings.Tune = (int)MathF.Round(values[7]);
        settings.A440 = values[8];
        settings.VelocityHardness = values[9] / 100;
        settings.VelocityMin = (int)MathF.Round(values[10]);
        settings.VelocityMax = (int)MathF.Round(values[11]);
        piano!.Apply(settings);
        piano.ControlsOn = values[4] >= .5f;
    }

    public int Render(uint* flags, void* time, uint bus, uint frames, AudioBufferList* io)
    {
        if (!initialized)
            return Status.Uninitialized;
        if (bus != 0)
            return Status.InvalidElement;
        if (frames > maxFrames)
            return Status.TooManyFrames;
        if (io == null || io->Count != 2)
            return Status.ParamError;
        var buffers = &io->First;
        for (var c = 0; c < 2; c++)
        {
            if (buffers[c].Data == null)
                buffers[c].Data = c == 0 ? left : right;
            else if (buffers[c].Bytes < frames * sizeof(float))
                return Status.ParamError;
            buffers[c].Bytes = frames * sizeof(float);
            buffers[c].Channels = 1;
        }
        var callbacks = notifies;
        var action = flags == null ? 0 : *flags;
        Notify(callbacks, action | 4, time, bus, frames, io);
        try
        {
            Produce(buffers[0].Data, buffers[1].Data, (int)frames);
        }
        catch (Exception)
        {
            lastError = Status.CannotDoNow;
            new Span<float>(buffers[0].Data, (int)frames).Clear();
            new Span<float>(buffers[1].Data, (int)frames).Clear();
        }
        Notify(callbacks, action | 8, time, bus, frames, io);
        return Status.Ok;
    }

    static void Notify((nint Proc, nint Data)[] callbacks, uint action, void* time, uint bus, uint frames, AudioBufferList* io)
    {
        foreach (var (proc, data) in callbacks)
            ((delegate* unmanaged<nint, uint*, void*, uint, uint, AudioBufferList*, int>)proc)(data, &action, time, bus, frames, io);
    }

    void Produce(float* l, float* r, int frames)
    {
        if (Interlocked.Exchange(ref reset, 0) != 0)
            Silence();
        Apply();
        Schedule();
        if (resampler is null)
        {
            Synthesize(scratch, frames);
            for (var n = 0; n < frames; n++)
            {
                l[n] = scratch[2 * n];
                r[n] = scratch[2 * n + 1];
            }
        }
        else
            resampler.Process(l, r, frames);
        var target = Gain();
        var step = (target - gain) / frames;
        for (var n = 0; n < frames; n++)
        {
            var g = gain + step * (n + 1);
            l[n] *= g;
            r[n] *= g;
        }
        gain = target;
        Array.Copy(schedule, head, schedule, 0, scheduled - head);
        scheduled -= head;
        head = 0;
    }

    void Silence()
    {
        piano!.Controller(64, 0);
        piano.Controller(120, 0);
        head = scheduled = 0;
        resampler?.Reset(written);
    }

    void Schedule()
    {
        int count;
        lock (gate)
        {
            count = incomingCount;
            Array.Copy(incoming, batch, count);
            incomingCount = 0;
        }
        var time = resampler?.Time ?? written;
        var step = resampler?.Step ?? 1;
        var delay = resampler?.Delay ?? 0;
        for (var i = 0; i < count && scheduled < Queue; i++)
        {
            var e = batch[i];
            var target = (long)Math.Ceiling(time + e.Offset * step) + delay;
            var j = scheduled++;
            for (; j > head && schedule[j - 1].Target > target; j--)
                schedule[j] = schedule[j - 1];
            schedule[j] = new Scheduled(target, e.Status, e.Data1, e.Data2);
        }
    }

    void Synthesize(float[] stereo, int count)
    {
        Array.Clear(stereo, 0, 2 * count);
        for (var done = 0; done < count;)
        {
            while (head < scheduled && schedule[head].Target <= written)
                Play(schedule[head++]);
            var chunk = count - done;
            if (head < scheduled)
                chunk = (int)Math.Min(chunk, schedule[head].Target - written);
            piano!.Render(stereo, done, chunk);
            done += chunk;
            written += chunk;
        }
    }

    void Play(Scheduled e)
    {
        switch (e.Status & 0xF0)
        {
            case 0x90:
                piano!.NoteOn(e.Data1 & 0x7F, e.Data2 & 0x7F);
                break;
            case 0x80:
                piano!.NoteOff(e.Data1 & 0x7F);
                break;
            case 0xB0 when e.Data1 is 64 or 120 or 123:
                piano!.Controller(e.Data1, e.Data2);
                break;
            case 0xB0 when e.Data1 == 121:
                piano!.Controller(64, 0);
                break;
        }
    }

    public int Midi(uint status, uint data1, uint data2, uint offset)
    {
        lock (gate)
            if (incomingCount < Queue)
                incoming[incomingCount++] = new Event((int)Math.Min(offset, 1u << 24), (byte)status, (byte)data1, (byte)data2);
        return Status.Ok;
    }

    public int Reset()
    {
        Interlocked.Exchange(ref reset, 1);
        return Status.Ok;
    }

    public int GetParameter(uint id, uint scope, float* value)
    {
        if (scope != Global)
            return Status.InvalidScope;
        if (id >= Parameters.Length)
            return Status.InvalidParameter;
        if (value == null)
            return Status.ParamError;
        *value = values[id];
        return Status.Ok;
    }

    public int SetParameter(uint id, uint scope, float value)
    {
        if (scope != Global)
            return Status.InvalidScope;
        if (id >= Parameters.Length)
            return Status.InvalidParameter;
        if (!float.IsFinite(value))
            return Status.InvalidParameterValue;
        values[id] = Math.Clamp(value, Parameters[id].Min, Parameters[id].Max);
        if (id != Volume)
            Interlocked.Increment(ref version);
        return Status.Ok;
    }

    public int ScheduleParameters(ParameterEvent* events, uint count)
    {
        for (var i = 0; i < count; i++)
        {
            var status = SetParameter(events[i].Parameter, events[i].Scope, events[i].Kind == 2 ? events[i].End : events[i].Value);
            if (status != Status.Ok)
                return status;
        }
        return Status.Ok;
    }

    public int AddListener(uint id, nint proc, nint data)
    {
        lock (gate)
            listeners = [.. listeners, (id, proc, data)];
        return Status.Ok;
    }

    public int RemoveListener(uint id, nint proc, nint data, bool any)
    {
        lock (gate)
            listeners = listeners.Where(l => l.Id != id || l.Proc != proc || !any && l.Data != data).ToArray();
        return Status.Ok;
    }

    public int AddNotify(nint proc, nint data)
    {
        lock (gate)
            notifies = [.. notifies, (proc, data)];
        return Status.Ok;
    }

    public int RemoveNotify(nint proc, nint data)
    {
        lock (gate)
            notifies = notifies.Where(n => n.Proc != proc || n.Data != data).ToArray();
        return Status.Ok;
    }

    void Changed(uint id, uint scope = Global)
    {
        foreach (var (listener, proc, data) in listeners)
            if (listener == id)
                ((delegate* unmanaged<nint, nint, uint, uint, uint, void>)proc)(data, handle, id, scope, 0);
    }

    enum Where { None, Global, Stream, Output, Parameter, Any }

    public int Info(uint id, uint scope, uint element, out uint size, out bool writable)
    {
        (size, writable) = (0, false);
        var (bytes, canWrite, where) = id switch
        {
            Property.ClassInfo => (8u, true, Where.Global),
            Property.SampleRate => (8u, true, Where.Stream),
            Property.ParameterList => (scope == Global ? 4u * (uint)Parameters.Length : 0u, false, Where.Any),
            Property.ParameterInfo => ((uint)sizeof(ParameterInfo), false, Where.Parameter),
            Property.StreamFormat => ((uint)sizeof(StreamFormat), true, Where.Output),
            Property.ElementCount => (4u, false, Where.Any),
            Property.Latency or Property.TailTime => (8u, false, Where.Global),
            Property.SupportedNumChannels or Property.LastRenderError or Property.InstrumentCount => (4u, false, Where.Global),
            Property.MaximumFramesPerSlice or Property.RenderQuality or Property.OfflineRender => (4u, true, Where.Global),
            Property.HostCallbacks => ((uint)hostCallbacks.Length, true, Where.Global),
            Property.PresentPreset => ((uint)sizeof(Preset), true, Where.Global),
            Property.ShouldAllocateBuffer => (4u, true, Where.Output),
            Property.ContextName or Property.NickName => (8u, true, Where.Global),
            _ => (0u, false, Where.None),
        };
        var status = where switch
        {
            Where.None => Status.InvalidProperty,
            Where.Global => scope != Global ? Status.InvalidScope : element != 0 ? Status.InvalidElement : Status.Ok,
            Where.Stream => scope == Input ? Status.InvalidElement : scope is not (Global or Output) ? Status.InvalidScope : element != 0 ? Status.InvalidElement : Status.Ok,
            Where.Output => scope == Input ? Status.InvalidElement : scope != Output ? Status.InvalidScope : element != 0 ? Status.InvalidElement : Status.Ok,
            Where.Parameter => scope != Global || element >= Parameters.Length ? Status.InvalidParameter : Status.Ok,
            _ => scope > Group ? Status.InvalidScope : Status.Ok,
        };
        if (status == Status.Ok)
            (size, writable) = (bytes, canWrite);
        return status;
    }

    static bool IsObject(uint id) => id is Property.ClassInfo or Property.PresentPreset or Property.ContextName or Property.NickName;

    public int GetProperty(uint id, uint scope, uint element, void* data, uint* size)
    {
        var status = Info(id, scope, element, out var needed, out _);
        if (status != Status.Ok)
            return status;
        if (size == null)
            return Status.ParamError;
        if (data == null)
        {
            *size = needed;
            return Status.Ok;
        }
        if (*size >= needed)
        {
            Get(id, scope, element, (byte*)data);
            *size = needed;
            return Status.Ok;
        }
        if (IsObject(id))
            return Status.ParamError;
        var temp = stackalloc byte[(int)needed];
        Get(id, scope, element, temp);
        Buffer.MemoryCopy(temp, data, *size, *size);
        return Status.Ok;
    }

    void Get(uint id, uint scope, uint element, byte* data)
    {
        switch (id)
        {
            case Property.ClassInfo:
                *(nint*)data = Save();
                break;
            case Property.SampleRate:
                *(double*)data = rate;
                break;
            case Property.ParameterList:
                if (scope == Global)
                    for (var i = 0u; i < Parameters.Length; i++)
                        ((uint*)data)[i] = i;
                break;
            case Property.ParameterInfo:
                Describe(element, (ParameterInfo*)data);
                break;
            case Property.StreamFormat:
                *(StreamFormat*)data = new StreamFormat
                {
                    SampleRate = rate, FormatId = Lpcm, FormatFlags = 1 | 8 | 32, BytesPerPacket = 4, FramesPerPacket = 1, BytesPerFrame = 4,
                    ChannelsPerFrame = 2, BitsPerChannel = 32,
                };
                break;
            case Property.ElementCount:
                *(uint*)data = scope is Global or Output ? 1u : 0u;
                break;
            case Property.Latency:
                *(double*)data = rate == ModalBank.Rate ? 0 : (Resampler.Taps(rate) / 2 + 1) / (double)ModalBank.Rate;
                break;
            case Property.TailTime:
                *(double*)data = 0;
                break;
            case Property.SupportedNumChannels:
                ((short*)data)[0] = 0;
                ((short*)data)[1] = 2;
                break;
            case Property.MaximumFramesPerSlice:
                *(uint*)data = maxFrames;
                break;
            case Property.LastRenderError:
                *(int*)data = lastError;
                lastError = 0;
                break;
            case Property.InstrumentCount:
                *(uint*)data = 0;
                break;
            case Property.RenderQuality:
                *(uint*)data = quality;
                break;
            case Property.OfflineRender:
                *(uint*)data = offline;
                break;
            case Property.ShouldAllocateBuffer:
                *(uint*)data = allocate;
                break;
            case Property.HostCallbacks:
                hostCallbacks.CopyTo(new Span<byte>(data, hostCallbacks.Length));
                break;
            case Property.PresentPreset:
                *(Preset*)data = new Preset { Number = presetNumber, Name = Retain(presetName) };
                break;
            case Property.ContextName:
                *(nint*)data = Retain(contextName);
                break;
            case Property.NickName:
                *(nint*)data = Retain(nickName);
                break;
        }
    }

    static nint Retain(nint value) => value == 0 ? 0 : CF.CFRetain(value);

    static void Replace(ref nint field, nint value)
    {
        Retain(value);
        if (field != 0)
            CF.CFRelease(field);
        field = value;
    }

    static void Describe(uint id, ParameterInfo* info)
    {
        var p = Parameters[id];
        *info = default;
        var name = Encoding.UTF8.GetBytes(p.Name);
        name.AsSpan(0, Math.Min(name.Length, 51)).CopyTo(new Span<byte>(info->Name, 52));
        info->CFName = names[id];
        info->Unit = p.Unit;
        (info->Min, info->Max, info->Default) = (p.Min, p.Max, p.Default);
        info->Flags = Readable | Writable | HasName;
    }

    public int SetProperty(uint id, uint scope, uint element, void* data, uint size)
    {
        var status = Info(id, scope, element, out var needed, out var writable);
        if (status != Status.Ok)
            return status;
        if (!writable)
            return Status.NotWritable;
        if (data == null || size < needed && id != Property.HostCallbacks)
            return Status.ParamError;
        switch (id)
        {
            case Property.ClassInfo:
                return Restore(*(nint*)data);
            case Property.SampleRate:
                return SetRate(*(double*)data);
            case Property.StreamFormat:
                var format = *(StreamFormat*)data;
                return Valid(format) ? SetRate(format.SampleRate) : Status.FormatNotSupported;
            case Property.MaximumFramesPerSlice:
                var frames = *(uint*)data;
                if (frames == 0)
                    return Status.InvalidValue;
                if (frames == maxFrames)
                    return Status.Ok;
                if (initialized)
                    return Status.Initialized;
                maxFrames = frames;
                Changed(id);
                return Status.Ok;
            case Property.RenderQuality:
                quality = *(uint*)data;
                return Status.Ok;
            case Property.OfflineRender:
                offline = *(uint*)data;
                return Status.Ok;
            case Property.ShouldAllocateBuffer:
                allocate = *(uint*)data;
                return Status.Ok;
            case Property.HostCallbacks:
                Array.Clear(hostCallbacks);
                new ReadOnlySpan<byte>(data, (int)Math.Min(size, (uint)hostCallbacks.Length)).CopyTo(hostCallbacks);
                return Status.Ok;
            case Property.PresentPreset:
                var preset = *(Preset*)data;
                presetNumber = preset.Number;
                Replace(ref presetName, preset.Name);
                Changed(id);
                return Status.Ok;
            case Property.ContextName:
                Replace(ref contextName, *(nint*)data);
                return Status.Ok;
            case Property.NickName:
                Replace(ref nickName, *(nint*)data);
                return Status.Ok;
        }
        return Status.InvalidProperty;
    }

    static bool Valid(StreamFormat f) =>
        f.FormatId == Lpcm && (f.FormatFlags & 1) != 0 && (f.FormatFlags & 32) != 0 && (f.FormatFlags & (2 | 4)) == 0 && f.BitsPerChannel == 32 &&
        f.ChannelsPerFrame == 2 && f.BytesPerFrame == 4 && f.BytesPerPacket == 4 && f.FramesPerPacket == 1;

    int SetRate(double value)
    {
        if (!(value > 0) || double.IsInfinity(value))
            return Status.InvalidValue;
        if (value == rate)
            return Status.Ok;
        if (initialized)
            return Status.Initialized;
        rate = value;
        Changed(Property.SampleRate, Output);
        Changed(Property.StreamFormat, Output);
        Changed(Property.Latency);
        return Status.Ok;
    }

    nint Save()
    {
        var dictionary = CF.Dictionary();
        CF.Set(dictionary, "version", CF.Number(0));
        CF.Set(dictionary, "type", CF.Number(unchecked((int)Type)));
        CF.Set(dictionary, "subtype", CF.Number(unchecked((int)Subtype)));
        CF.Set(dictionary, "manufacturer", CF.Number(unchecked((int)Manufacturer)));
        CF.Set(dictionary, "data", CF.Data(MemoryMarshal.AsBytes(values.AsSpan())));
        CF.Set(dictionary, "name", presetName != 0 ? CF.CFRetain(presetName) : CF.String("Untitled"));
        return dictionary;
    }

    int Restore(nint dictionary)
    {
        if (!CF.IsDictionary(dictionary) || CF.Int(CF.Get(dictionary, "version")) != 0 || CF.Int(CF.Get(dictionary, "type")) != unchecked((int)Type) ||
            CF.Int(CF.Get(dictionary, "subtype")) != unchecked((int)Subtype) || CF.Int(CF.Get(dictionary, "manufacturer")) != unchecked((int)Manufacturer))
            return Status.InvalidValue;
        var stored = MemoryMarshal.Cast<byte, float>(CF.Bytes(CF.Get(dictionary, "data")));
        for (var i = 0; i < Math.Min(stored.Length, values.Length); i++)
            if (float.IsFinite(stored[i]))
                values[i] = Math.Clamp(stored[i], Parameters[i].Min, Parameters[i].Max);
        var name = CF.Get(dictionary, "name");
        if (CF.IsString(name))
            Replace(ref presetName, name);
        presetNumber = -1;
        Interlocked.Increment(ref version);
        for (var i = 0u; i < Parameters.Length; i++)
        {
            var parameter = new UnitParameter { Unit = handle, Id = i };
            Native.AUParameterListenerNotify(0, 0, &parameter);
        }
        Changed(Property.PresentPreset);
        return Status.Ok;
    }
}
