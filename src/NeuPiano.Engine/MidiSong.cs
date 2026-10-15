using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;

namespace NeuPiano.Engine;

public enum MidiKind { NoteOn, NoteOff, Control }

public readonly record struct MidiEvent(long Sample, MidiKind Kind, int Note, int Value);

public sealed class MidiSong
{
    public string Path { get; }
    public IReadOnlyList<MidiEvent> Events { get; }
    public double Seconds => Events.Count == 0 ? 0 : (double)Events[^1].Sample / ModalBank.Rate;

    MidiSong(string path, IReadOnlyList<MidiEvent> events) => (Path, Events) = (path, events);

    public static MidiSong Load(string path)
    {
        var file = MidiFile.Read(path);
        var tempo = file.GetTempoMap();
        var events = new List<MidiEvent>();
        foreach (var timed in file.GetTimedEvents())
        {
            if (timed.Event is ChannelEvent channel && channel.Channel == 9)
                continue;
            var sample = timed.TimeAs<MetricTimeSpan>(tempo).TotalMicroseconds * ModalBank.Rate / 1_000_000;
            MidiEvent? midi = timed.Event switch
            {
                NoteOnEvent on when on.Velocity > 0 => new(sample, MidiKind.NoteOn, on.NoteNumber, on.Velocity),
                NoteOnEvent on => new(sample, MidiKind.NoteOff, on.NoteNumber, 0),
                NoteOffEvent off => new(sample, MidiKind.NoteOff, off.NoteNumber, 0),
                ControlChangeEvent control => new(sample, MidiKind.Control, control.ControlNumber, control.ControlValue),
                _ => null,
            };
            if (midi is not null)
                events.Add(midi.Value);
        }
        var start = events.Where(e => e.Kind == MidiKind.NoteOn).Select(e => e.Sample).DefaultIfEmpty().Min();
        return new MidiSong(path, events
            .Select(e => e with { Sample = Math.Max(0, e.Sample - start) })
            .OrderBy(e => e.Sample)
            .ToList());
    }
}
