using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using NeuPiano.Engine;

namespace NeuPiano.App;

public partial class MainWindow : Window
{
    const string Modal = "modal_bank.npz", Control = "control_net.npz", Releases = "release_bank.npz";
    readonly Synth synth = new();
    readonly AudioOutput? audio;
    MidiSong? song;
    string modal = $"Modal bank not found: put {Modal} in a banks folder next to the app or one of its parents", message = "";
    ModalPiano? modalPiano;
    int frame;

    public MainWindow() : this([]) { }

    public MainWindow(string[] args)
    {
        InitializeComponent();
        Keyboard.Synth = synth;
        VolumeSlider.Value = 20 * Math.Log10(synth.Volume);
        try
        {
            audio = new AudioOutput(synth);
        }
        catch (Exception e)
        {
            message = $"Audio unavailable: {e.Message}";
        }
        Closed += (_, _) => audio?.Dispose();
        var banks = args.FirstOrDefault(Directory.Exists);
        if ((banks is null ? FindUpwards(Modal) : Existing(Path.Combine(banks, Modal))) is { } modalPath)
            _ = LoadModal(modalPath);
        if (args.FirstOrDefault(a => a.EndsWith(".mid")) is { } midiPath)
            _ = LoadMidi(midiPath);
        DispatcherTimer.Run(Refresh, TimeSpan.FromMilliseconds(33));
    }

    static string? Existing(string path) => File.Exists(path) ? path : null;

    static string? FindUpwards(string name)
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
            foreach (var path in new[] { Path.Combine(directory.FullName, name), Path.Combine(directory.FullName, "banks", name) })
                if (File.Exists(path))
                    return path;
        return null;
    }

    async Task LoadModal(string path)
    {
        modal = "Modal synth loading...";
        var folder = Path.GetDirectoryName(path)!;
        var controlPath = Existing(Path.Combine(folder, Control));
        var releasePath = Existing(Path.Combine(folder, Releases));
        var controlsOn = ControlBox.IsChecked == true;
        try
        {
            var piano = await Task.Run(() =>
            {
                var controls = controlPath is null ? null : ControlNet.Load(controlPath);
                var releases = releasePath is null ? null : ModalBank.Load(releasePath);
                var value = new ModalPiano(ModalBank.Load(path), controls, releases) { ControlsOn = controlsOn };
                synth.LoadModal(value);
                return value;
            });
            modalPiano = piano;
            ControlBox.IsEnabled = piano.HasControls;
            ReleaseBox.IsEnabled = piano.HasReleases;
            Keyboard.Available = Enumerable.Range(piano.Bank.FirstNote, piano.Bank.Keys).ToHashSet();
            modal = $"Modal synth: {piano.Bank.Keys} keys, {piano.Bank.Partials} string partials" +
                    (piano.HasControls ? ", 50 Hz control net" : ", no control net") + (piano.HasReleases ? ", release sounds" : ", no release sounds");
        }
        catch (Exception e)
        {
            modal = "Modal synth unavailable";
            message = $"Could not load the modal synth: {e.Message}";
        }
    }

    async Task LoadMidi(string path)
    {
        try
        {
            song = await Task.Run(() => MidiSong.Load(path));
        }
        catch (Exception e)
        {
            message = $"Could not load MIDI file: {e.Message}";
        }
    }

    async void OpenMidi(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            FileTypeFilter = [new FilePickerFileType("MIDI file") { Patterns = ["*.mid"] }],
        });
        if (files.FirstOrDefault()?.TryGetLocalPath() is { } path)
            await LoadMidi(path);
    }

    void Play(object? sender, RoutedEventArgs e)
    {
        if (song is not null)
            synth.Play(song);
    }

    void Stop(object? sender, RoutedEventArgs e) => synth.StopSong();

    void SustainChanged(object? sender, RoutedEventArgs e) => synth.Sustain(SustainBox.IsChecked == true);

    void ControlChanged(object? sender, RoutedEventArgs e)
    {
        if (modalPiano is not null)
            modalPiano.ControlsOn = ControlBox.IsChecked == true;
    }

    void ReleaseChanged(object? sender, RoutedEventArgs e)
    {
        if (modalPiano is not null)
            modalPiano.ReleasesOn = ReleaseBox.IsChecked == true;
    }

    void VolumeChanged(object? sender, RoutedEventArgs e) => synth.Volume = MathF.Pow(10, (float)VolumeSlider.Value / 20);

    bool Refresh()
    {
        Keyboard.InvalidateVisual();
        if (frame++ % 8 != 0)
            return true;
        PlayButton.IsEnabled = song is not null && !synth.Playing;
        var midi = song is null ? "no MIDI file"
            : $"{Path.GetFileName(song.Path)} {(synth.Playing ? synth.SongSeconds : 0):0.0}/{song.Seconds:0.0} s";
        Status.Text = $"{modal}  |  {midi}  |  last velocity {Keyboard.LastVelocity}  |  voices {synth.VoiceCount}" +
                      $"  |  render load {synth.CpuLoad:P0}  |  {audio?.DeviceName ?? "no audio"}: buffer {audio?.LatencyMs ?? 0:0} ms, dropouts {audio?.Dropouts ?? 0}  {message}";
        return true;
    }
}
