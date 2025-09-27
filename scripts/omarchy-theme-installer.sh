#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Theme database with correct URLs from awesome-omarchy
declare -A THEMES=(
  ["Alabaster"]="https://github.com/grierson/omarchy-alabaster-theme"
  ["All Hallows Eve"]="https://github.com/guilhermetk/omarchy-all-hallows-eve-theme"
  ["Anonymous"]="https://github.com/j4v3l/omarchy-anonymous-theme"
  ["Ash"]="https://github.com/bjarneo/omarchy-ash-theme"
  ["Aura"]="https://github.com/bjarneo/omarchy-aura-theme"
  ["Ayaka"]="https://github.com/abhijeet-swami/omarchy-ayaka-theme"
  ["Ayu Dark"]="https://github.com/fdidron/omarchy-ayu-dark-theme"
  ["Ayu Light"]="https://github.com/fdidron/omarchy-ayu-light-theme"
  ["Ayu Mirage"]="https://github.com/fdidron/omarchy-ayu-mirage-theme"
  ["Azure Glow"]="https://github.com/Hydradevx/omarchy-azure-glow-theme"
  ["Bauhaus"]="https://github.com/somerocketeer/omarchy-bauhaus-theme"
  ["Blackgold"]="https://github.com/HANCORE-linux/omarchy-blackgold-theme"
  ["Blackturq"]="https://github.com/HANCORE-linux/omarchy-blackturq-theme"
  ["Bluedotrb"]="https://github.com/dotsilva/omarchy-bluedotrb-theme"
  ["Blueridge Dark"]="https://github.com/hipsterusername/omarchy-blueridge-dark-theme"
  ["Cobalt2"]="https://github.com/hoblin/omarchy-cobalt2-theme"
  ["Crimson Gold"]="https://github.com/knappkevin/omarchy-crimson-gold-theme"
  ["Dotrb"]="https://github.com/dotsilva/omarchy-dotrb-theme"
  ["Dracula"]="https://github.com/catlee/omarchy-dracula-theme"
  ["Everblush"]="https://github.com/dfrico/omarchy-solarized-light-theme"
  ["Ember N Ash"]="https://github.com/Hydradevx/omarchy-ember-n-ash-theme"
  ["Felix"]="https://github.com/TyRichards/omarchy-felix-theme"
  ["Fiery Ocean"]="https://github.com/bjarneo/omarchy-fiery-ocean-theme"
  ["Fireside"]="https://github.com/bjarneo/omarchy-fireside-theme"
  ["Firesky"]="https://github.com/bjarneo/omarchy-firesky-theme"
  ["Flexoki Dark"]="https://github.com/euandeas/omarchy-flexoki-dark-theme"
  ["Flexoki Light"]="https://github.com/euandeas/omarchy-flexoki-light-theme"
  ["Forest Green"]="https://github.com/abhijeet-swami/omarchy-forest-green-theme"
  ["Frost"]="https://github.com/bjarneo/omarchy-frost-theme"
  ["Futurism"]="https://github.com/bjarneo/omarchy-futurism-theme"
  ["Gold Rush"]="https://github.com/tahayvr/omarchy-gold-rush-theme"
  ["Green Garden"]="https://github.com/kalk-ak/omarchy-green-garden-theme"
  ["Hakker Green"]="https://github.com/joaquinmeza/omarchy-hakker-green-theme"
  ["Hollow Knight"]="https://github.com/bjarneo/omarchy-hollow-knight-theme"
  ["Kimiko"]="https://github.com/krymzonn/omarchy-kimiko-theme"
  ["Komorebi"]="https://github.com/ryuhzk/komorebi"
  ["Mars"]="https://github.com/steve-lohmeyer/omarchy-mars-theme"
  ["Memento Mori"]="https://github.com/hipsterusername/omarchy-memento-mori-theme"
  ["Midnight"]="https://github.com/JaxonWright/omarchy-midnight-theme"
  ["Milkmatcha Light"]="https://github.com/hipsterusername/omarchy-milkmatcha-light-theme"
  ["Monochrome"]="https://github.com/Swarnim114/omarchy-monochrome-theme"
  ["Monokai"]="https://github.com/bjarneo/omarchy-monokai-theme"
  ["Monokai Dark"]="https://github.com/ericrswanny/omarchy-monokai-dark-theme"
  ["Nagai Poolside"]="https://github.com/somerocketeer/omarchy-nagai-poolside-theme"
  ["Nagai Twilight"]="https://github.com/somerocketeer/omarchy-nagai-twilight-theme"
  ["NES"]="https://github.com/bjarneo/omarchy-nes-theme"
  ["One Dark Pro"]="https://github.com/sc0ttman/omarchy-one-dark-pro"
  ["Osaka Jade"]="https://github.com/Justikun/omarchy-osaka-jade-theme"
  ["Pina"]="https://github.com/bjarneo/omarchy-pina-theme"
  ["Pretty CVNT"]="https://github.com/WalkerMillgress/omarchy-pretty-cvnt-theme"
  ["Pulsar"]="https://github.com/bjarneo/omarchy-pulsar-theme"
  ["Purplewave"]="https://github.com/dotsilva/omarchy-purplewave-theme"
  ["RetroPC"]="https://github.com/rondilley/omarchy-retropc-theme"
  ["Rose Pine Dark"]="https://github.com/guilhermetk/omarchy-rose-pine-dark"
  ["Sakura"]="https://github.com/bjarneo/omarchy-sakura-theme"
  ["Serenity"]="https://github.com/bjarneo/omarchy-serenity-theme"
  ["Snow"]="https://github.com/bjarneo/omarchy-snow-theme"
  ["Solarized"]="https://github.com/Gazler/omarchy-solarized-theme"
  ["Solarized Light"]="https://github.com/dfrico/omarchy-solarized-light-theme"
  ["Solarized Osaka"]="https://github.com/motorsss/omarchy-solarizedosaka-theme"
  ["Space Monkey"]="https://github.com/TyRichards/omarchy-space-monkey-theme"
  ["Super Game Bro"]="https://github.com/TyRichards/omarchy-super-game-bro-theme"
  ["Synthwave '84"]="https://github.com/omacom-io/omarchy-synthwave84-theme"
  ["Tekk-o-ween"]="https://github.com/joaquinmeza/omarchy-tekk-o-ween-theme"
  ["Vague"]="https://github.com/Rnedlose/omarchy-vague-theme"
  ["Vague Light"]="https://github.com/Rnedlose/omarchy-vague-light-theme"
  ["Velocity"]="https://github.com/perfektnacht/omarchy-velocity-theme"
  ["Vercel"]="https://github.com/somerocketeer/omarchy-vercel-theme"
  ["Vesper"]="https://github.com/RuanAlbertoSilva/omarchy-vesper-theme"
  ["VHS80"]="https://github.com/tahayvr/omarchy-vhs80-theme"
  ["Vice City"]="https://github.com/lavarinimoreira/omarchy-vice-city-theme"
  ["Void"]="https://github.com/vyrx-dev/omarchy-void-theme"
  ["Wasteland"]="https://github.com/perfektnacht/omarchy-wasteland-theme"
  ["Waveform Dark"]="https://github.com/hipsterusername/omarchy-waveform-dark-theme"
)

# Check dependencies
if ! command -v gum &>/dev/null; then
  echo -e "${YELLOW}Error: gum not found. Install with: pacman -S gum${NC}"
  exit 1
fi

if ! command -v omarchy-theme-install &>/dev/null; then
  echo -e "${YELLOW}Error: omarchy-theme-install not found${NC}"
  exit 1
fi

# Main installation loop
while true; do
  # Show header
  clear
  TERM_WIDTH=$(tput cols)
  BOX_LINES=(
    "╔════════════════════════════════════════╗"
    "║         OMARCHY THEME INSTALLER        ║"
    "╚════════════════════════════════════════╝"
    "Bro"
  )
  for line in "${BOX_LINES[@]}"; do
    padding=$(((TERM_WIDTH - ${#line}) / 2))
    printf "%*s%s\n" "$padding" "" "${CYAN}${line}${NC}"
  done
  echo
  echo -e "${BLUE}Choose your usdefault theme${NC}"
  echo -e "${YELLOW}Navigate with ↑ ↓ arrows, press Enter to select${NC}"
  echo

  # Get user selection using gum filter
  selected=$(printf '%s\n' "${!THEMES[@]}" | sort | gum filter --placeholder="Search themes...")

  if [ -z "$selected" ]; then
    echo
    echo -e "${YELLOW}No theme selected${NC}"
    exit 0
  fi

  # Confirmation dialog
  echo
  if ! gum confirm "Install theme: $selected?"; then
    echo -e "${YELLOW}Installation cancelled${NC}"
    continue
  fi

  # Install selected theme
  url="${THEMES[$selected]}"
  echo
  echo -e "${BLUE}Installing theme:${NC} ${PURPLE}$selected${NC}"
  echo -e "${YELLOW}URL:${NC} $url"
  echo

  if gum spin --spinner dot --title "Installing theme..." -- omarchy-theme-install "$url"; then
    echo
    echo -e "${GREEN}✓ Theme '$selected' installed successfully!${NC}"
    echo -e "${CYAN}You can now select it via: Super + Ctrl + Shift + Space${NC}"
  else
    echo
    echo -e "${YELLOW}✗ Failed to install theme '$selected'${NC}"
  fi

  echo
  # Ask if user wants to install another theme
  if ! gum confirm "Install another theme?"; then
    break
  fi
done

echo
echo -e "${GREEN}Theme installation complete!${NC}"

