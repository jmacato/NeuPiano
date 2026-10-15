using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using NeuPiano.Engine;
using PortAudioSharp;
using Stream = PortAudioSharp.Stream;

namespace NeuPiano.App;

public sealed class AudioOutput : IDisposable
{
    public const int Block = 512;
    readonly Synth synth;
    readonly Stream.Callback callback;
    readonly float[] stereo = new float[2 * Synth.MaxFrames];
    readonly object gate = new();
    readonly IDisposable? watcher;
    Stream? stream;
    uint device;
    double latency;
    long dropouts;
    bool disposed;

    public long Dropouts => Interlocked.Read(ref dropouts);
    public double LatencyMs => latency * 1000;
    public string DeviceName { get; private set; } = "no output device";

    public AudioOutput(Synth synth)
    {
        this.synth = synth;
        callback = Render;
        PortAudio.Initialize();
        Open();
        if (OperatingSystem.IsMacOS())
            watcher = new DefaultOutputWatcher(Follow);
    }

    void Open()
    {
        if (OperatingSystem.IsMacOS())
            device = DefaultOutputWatcher.Current();
        var index = PortAudio.DefaultOutputDevice;
        if (index == PortAudio.NoDevice)
        {
            DeviceName = "no output device";
            return;
        }
        var info = PortAudio.GetDeviceInfo(index);
        var output = new StreamParameters
        {
            device = index,
            channelCount = 2,
            sampleFormat = SampleFormat.Float32,
            suggestedLatency = info.defaultLowOutputLatency,
        };
        latency = output.suggestedLatency + (double)Block / ModalBank.Rate;
        stream = new Stream(null, output, ModalBank.Rate, Block, StreamFlags.ClipOff, callback, null);
        stream.Start();
        DeviceName = info.name;
    }

    void Close()
    {
        stream?.Stop();
        stream?.Dispose();
        stream = null;
    }

    void Follow()
    {
        lock (gate)
        {
            if (disposed || OperatingSystem.IsMacOS() && DefaultOutputWatcher.Current() == device && stream is not null)
                return;
            try
            {
                Close();
                PortAudio.Terminate();
                PortAudio.Initialize();
                Open();
            }
            catch (Exception e)
            {
                stream = null;
                DeviceName = $"output unavailable: {e.Message}";
            }
        }
    }

    StreamCallbackResult Render(IntPtr input, IntPtr output, uint frameCount, ref StreamCallbackTimeInfo timeInfo,
                                StreamCallbackFlags statusFlags, IntPtr userData)
    {
        var started = Stopwatch.GetTimestamp();
        var frames = (int)Math.Min(frameCount, Synth.MaxFrames);
        try
        {
            synth.Render(stereo, frames);
        }
        catch (Exception)
        {
            Array.Clear(stereo);
            Interlocked.Increment(ref dropouts);
        }
        Marshal.Copy(stereo, 0, output, 2 * frames);
        if ((statusFlags & StreamCallbackFlags.OutputUnderflow) != 0
            || Stopwatch.GetElapsedTime(started).TotalSeconds * ModalBank.Rate > frames)
            Interlocked.Increment(ref dropouts);
        return StreamCallbackResult.Continue;
    }

    public void Dispose()
    {
        watcher?.Dispose();
        lock (gate)
        {
            disposed = true;
            Close();
            PortAudio.Terminate();
        }
    }
}
