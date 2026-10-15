using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using NeuPiano.Engine;

namespace NeuPiano.App;

sealed class PianoKeyboard : Control
{
    const int Lowest = 21, Highest = 108;
    static readonly int[] Notes = Enumerable.Range(Lowest, Highest - Lowest + 1).ToArray();
    static readonly int[] BlackPitches = [1, 3, 6, 8, 10];
    static readonly int[] Whites = Notes.Where(IsWhite).ToArray(), Blacks = Notes.Where(n => !IsWhite(n)).ToArray();
    static readonly int[] HitOrder = [.. Blacks, .. Whites];
    static readonly IBrush Pressed = new SolidColorBrush(Color.Parse("#3B82F6"));
    static readonly IBrush Ebony = new SolidColorBrush(Color.Parse("#1E1E1E"));
    static readonly IBrush MissingWhite = new SolidColorBrush(Color.Parse("#C8C8C8"));
    static readonly IBrush MissingEbony = new SolidColorBrush(Color.Parse("#6E6E6E"));
    static readonly IPen Outline = new Pen(Brushes.Gray);
    static readonly Dictionary<int, FormattedText> Labels = Notes.Where(n => n % 12 == 0).ToDictionary(n => n, n =>
        new FormattedText($"C{n / 12 - 1}", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Inter"), 10, Brushes.Gray));
    readonly Rect[] rects = new Rect[Highest + 1];
    Size layout;
    int? held;

    public Synth? Synth { get; set; }
    public IReadOnlySet<int> Available { get; set; } = new HashSet<int>();
    public int LastVelocity { get; private set; }

    static bool IsWhite(int note) => !BlackPitches.Contains(note % 12);

    Rect KeyRect(int note)
    {
        if (layout != Bounds.Size)
        {
            layout = Bounds.Size;
            var width = Bounds.Width / Whites.Length;
            var whitesBefore = 0;
            foreach (var n in Notes)
            {
                rects[n] = IsWhite(n)
                    ? new Rect(whitesBefore++ * width, 0, width, Bounds.Height)
                    : new Rect((whitesBefore - .3) * width, 0, .6 * width, .62 * Bounds.Height);
            }
        }
        return rects[note];
    }

    public override void Render(DrawingContext context)
    {
        foreach (var note in Whites)
            DrawKey(context, note, Brushes.White, MissingWhite);
        foreach (var note in Blacks)
            DrawKey(context, note, Ebony, MissingEbony);
    }

    void DrawKey(DrawingContext context, int note, IBrush trained, IBrush raw)
    {
        var rect = KeyRect(note);
        context.DrawRectangle(Synth?.IsDown(note) == true ? Pressed : Available.Contains(note) ? trained : raw, Outline, rect);
        if (Labels.TryGetValue(note, out var label))
            context.DrawText(label, new Point(rect.X + 3, rect.Bottom - 16));
    }

    int? NoteAt(Point point)
    {
        foreach (var note in HitOrder)
            if (KeyRect(note).Contains(point))
                return note;
        return null;
    }

    void Strike(Point point)
    {
        held = NoteAt(point);
        if (held is not int note)
            return;
        var rect = KeyRect(note);
        LastVelocity = Math.Clamp((int)Math.Round(20 + 107 * (point.Y - rect.Y) / rect.Height), 1, 127);
        Synth?.NoteOn(note, LastVelocity);
    }

    void Lift()
    {
        if (held is int note)
            Synth?.NoteOff(note);
        held = null;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        e.Pointer.Capture(this);
        Strike(e.GetPosition(this));
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        if (held is not null && NoteAt(e.GetPosition(this)) != held)
        {
            Lift();
            Strike(e.GetPosition(this));
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e) => Lift();
}
