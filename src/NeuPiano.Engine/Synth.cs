using System.Collections.Concurrent;
using System.Diagnostics;

namespace NeuPiano.Engine;

public sealed class Synth
{
    public const int MaxFrames = 4096;

    readonly ConcurrentQueue<Action> commands = new();
    readonly MixBus bus = new();
    readonly bool[] down = new bool[128];
    PianoSettings? settings;
    ModalPiano? modal;
    MidiSong? song;
    int songIndex;
    long songSample;

    public float Volume { get; set; } = .25f;
    public double CpuLoad { get; private set; }
    public int VoiceCount { get; private set; }
    public bool Playing => song is not null;
    public double SongSeconds => (double)songSample / ModalBank.Rate;

    public bool IsDown(int note) => down[note];

    public void LoadModal(ModalPiano value)
    {
        var buffer = new float[2 * MaxFrames];
        value.NoteOn(21, 100);
        value.NoteOn(60, 100);
        value.Render(buffer, 0, MaxFrames);
        value.Controller(120, 0);
        commands.Enqueue(() =>
        {
            modal = value;
            if (settings is { } current)
                value.Apply(current);
        });
    }

    public void Apply(PianoSettings value)
    {
        var copy = value.Clone();
        commands.Enqueue(() =>
        {
            settings = copy;
            modal?.Apply(copy);
        });
    }

    public void NoteOn(int note, int velocity) => commands.Enqueue(() => Start(note, velocity));
    public void NoteOff(int note) => commands.Enqueue(() => Stop(note));
    public void Sustain(bool on) => commands.Enqueue(() => Control(64, on ? 127 : 0));
    public void Play(MidiSong value) => commands.Enqueue(() => { AllOff(); song = value; songIndex = 0; songSample = 0; });
    public void StopSong() => commands.Enqueue(() => { song = null; AllOff(); });

    public void Render(float[] stereo, int frames)
    {
        var started = Stopwatch.GetTimestamp();
        while (commands.TryDequeue(out var command))
            command();
        for (var done = 0; done < frames;)
        {
            var count = Math.Min(frames - done, MaxFrames);
            if (song is not null)
            {
                while (songIndex < song.Events.Count && song.Events[songIndex].Sample <= songSample)
                    Apply(song.Events[songIndex++]);
                if (songIndex < song.Events.Count)
                    count = (int)Math.Min(count, song.Events[songIndex].Sample - songSample);
                else if ((modal?.ActiveVoices ?? 0) == 0)
                    song = null;
            }
            Array.Clear(stereo, 2 * done, 2 * count);
            modal?.Render(stereo, done, count);
            done += count;
            songSample += song is null ? 0 : count;
        }
        bus.Process(stereo, frames, Volume);
        VoiceCount = modal?.ActiveVoices ?? 0;
        var load = Stopwatch.GetElapsedTime(started).TotalSeconds * ModalBank.Rate / frames;
        CpuLoad += .05 * (load - CpuLoad);
    }

    void Apply(MidiEvent e)
    {
        switch (e.Kind)
        {
            case MidiKind.NoteOn: Start(e.Note, e.Value); break;
            case MidiKind.NoteOff: Stop(e.Note); break;
            case MidiKind.Control: Control(e.Note, e.Value); break;
        }
    }

    void Control(int number, int value) => modal?.Controller(number, value);

    void Start(int note, int velocity)
    {
        down[note] = true;
        modal?.NoteOn(note, velocity);
    }

    void Stop(int note)
    {
        down[note] = false;
        modal?.NoteOff(note);
    }

    void AllOff()
    {
        Array.Clear(down);
        modal?.Controller(123, 0);
    }
}
