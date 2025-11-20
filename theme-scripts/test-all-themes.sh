#!/bin/bash

SCRIPT_DIR="$HOME/dotfiles/theme-scripts"
THEMES_DIR="$HOME/dotfiles/themes"
INTERVAL=2
TIMEOUT=10

echo "Testing themes (Ctrl+C to stop)"
sleep 1

STATIC_THEMES=(
    "aamis"
    "forest"
    "gruvbox-material"
    "rose-pine"
    "sakura"
    "tokyo-night"
    "vague"
    "void"
    "zen"
)

# Test static themes
for theme in "${STATIC_THEMES[@]}"; do
    if [ -d "$THEMES_DIR/$theme" ]; then
        "$SCRIPT_DIR/symphony-theme" switch "$theme" >/dev/null 2>&1
        echo "✓ $theme"
        sleep "$INTERVAL"
    fi
done

# Test matugen with change-theme script
"$SCRIPT_DIR/symphony-theme" switch "matugen" >/dev/null 2>&1
echo "✓ matugen"
sleep "$INTERVAL"

for i in {1..5}; do
    # Run change-theme with timeout to prevent hanging
    timeout "$TIMEOUT" "$HOME/.config/hypr/scripts/change-theme" >/dev/null 2>&1 &
    THEME_PID=$!
    wait $THEME_PID 2>/dev/null
    echo "✓ matugen #$i"
    sleep "$INTERVAL"
done

echo "Done!"
