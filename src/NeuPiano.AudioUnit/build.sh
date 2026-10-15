#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
banks="${BANKS:-$(cd "$here/../.." && pwd)/banks}"
bundle="$here/bin/NeuPiano.component"

dotnet publish "$here/NeuPiano.AudioUnit.csproj" -c Release -o "$here/bin/publish"
rm -rf "$bundle"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
cp "$here/Info.plist" "$bundle/Contents/"
cp "$here/bin/publish/NeuPiano.dylib" "$bundle/Contents/MacOS/NeuPiano"
cp "$banks/modal_bank.npz" "$bundle/Contents/Resources/"
for optional in control_net.npz release_bank.npz; do
    if [[ -f "$banks/$optional" ]]; then cp "$banks/$optional" "$bundle/Contents/Resources/"; fi
done
codesign --force --sign - "$bundle"

if [[ "${1:-}" == install ]]; then
    target="$HOME/Library/Audio/Plug-Ins/Components"
    mkdir -p "$target"
    rm -rf "$target/NeuPiano.component"
    cp -R "$bundle" "$target/"
    killall -9 AudioComponentRegistrar 2>/dev/null || true
    echo "installed $target/NeuPiano.component"
fi

if [[ "${1:-}" == package ]]; then
    stage="$here/bin/package"
    version=$(plutil -extract CFBundleShortVersionString raw "$here/Info.plist")
    rm -rf "$stage" "$here/bin/NeuPiano.zip"
    mkdir -p "$stage/root" "$stage/NeuPiano"
    ditto --norsrc --noextattr --noqtn --noacl "$bundle" "$stage/root/NeuPiano.component"
    COPYFILE_DISABLE=1 pkgbuild --analyze --root "$stage/root" "$stage/components.plist"
    plutil -replace 0.BundleIsRelocatable -bool NO "$stage/components.plist"
    plutil -replace 0.BundleIsVersionChecked -bool NO "$stage/components.plist"
    COPYFILE_DISABLE=1 pkgbuild --root "$stage/root" --component-plist "$stage/components.plist" --scripts "$here/installer/install" \
        --identifier local.neupiano.audiounit --version "$version" \
        --install-location /Library/Audio/Plug-Ins/Components "$stage/NeuPiano/Install NeuPiano.pkg"
    COPYFILE_DISABLE=1 pkgbuild --nopayload --scripts "$here/installer/uninstall" \
        --identifier local.neupiano.uninstall --version "$version" "$stage/NeuPiano/Uninstall NeuPiano.pkg"
    ditto -c -k --norsrc --noextattr --noqtn --keepParent "$stage/NeuPiano" "$here/bin/NeuPiano.zip"
    echo "packaged $here/bin/NeuPiano.zip"
fi
