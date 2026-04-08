############################
# 安装 zsh & oh-my-zsh
############################

install_zsh_env() {
  # 自动检测目标用户：优先使用 SUDO_USER，其次使用当前登录用户
  local TARGET_USER="${SUDO_USER:-$(id -un)}"

  # 自动获取用户 home 目录（兼容非 /home/ 前缀）
  local USER_HOME
  USER_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
  if [[ -z "${USER_HOME}" ]]; then
    USER_HOME="/home/${TARGET_USER}"
  fi

  local ZSHRC="${USER_HOME}/.zshrc"

  echo -e "${BLUE}[INFO]${NC} 安装 zsh 及常用工具..."
  apt update -y
  apt install -y zsh git curl autojump fonts-powerline

  echo -e "${BLUE}[INFO]${NC} 安装 oh-my-zsh（如未安装）..."
  if [[ ! -d "${USER_HOME}/.oh-my-zsh" ]]; then
    sudo -u "${TARGET_USER}" sh -c \
      'RUNZSH=no CHSH=no KEEP_ZSHRC=yes \
       sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"' || true
  else
    echo -e "${GREEN}[OK]${NC} oh-my-zsh 已存在，跳过安装"
  fi

  echo -e "${BLUE}[INFO]${NC} 安装 powerlevel10k 主题..."
  local P10K_DIR="${USER_HOME}/.oh-my-zsh/custom/themes/powerlevel10k"
  if [[ ! -d "${P10K_DIR}" ]]; then
    sudo -u "${TARGET_USER}" git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${P10K_DIR}"
  else
    echo -e "${GREEN}[OK]${NC} powerlevel10k 已存在，尝试更新"
    sudo -u "${TARGET_USER}" git -C "${P10K_DIR}" pull --ff-only || true
  fi

  echo -e "${BLUE}[INFO]${NC} 安装 zsh 插件（autosuggestions / syntax-highlighting / autojump 等）..."
  local PLUGIN_DIR="${USER_HOME}/.oh-my-zsh/custom/plugins"

  mkdir -p "${PLUGIN_DIR}"

  # zsh-autosuggestions
  if [[ ! -d "${PLUGIN_DIR}/zsh-autosuggestions" ]]; then
    sudo -u "${TARGET_USER}" git clone https://github.com/zsh-users/zsh-autosuggestions "${PLUGIN_DIR}/zsh-autosuggestions"
  else
    sudo -u "${TARGET_USER}" git -C "${PLUGIN_DIR}/zsh-autosuggestions" pull --ff-only || true
  fi

  # zsh-syntax-highlighting
  if [[ ! -d "${PLUGIN_DIR}/zsh-syntax-highlighting" ]]; then
    sudo -u "${TARGET_USER}" git clone https://github.com/zsh-users/zsh-syntax-highlighting "${PLUGIN_DIR}/zsh-syntax-highlighting"
  else
    sudo -u "${TARGET_USER}" git -C "${PLUGIN_DIR}/zsh-syntax-highlighting" pull --ff-only || true
  fi

  # 可选：zsh-autocomplete
  if [[ ! -d "${PLUGIN_DIR}/zsh-autocomplete" ]]; then
    sudo -u "${TARGET_USER}" git clone --depth=1 https://github.com/marlonrichert/zsh-autocomplete "${PLUGIN_DIR}/zsh-autocomplete" || true
  else
    sudo -u "${TARGET_USER}" git -C "${PLUGIN_DIR}/zsh-autocomplete" pull --ff-only || true
  fi

  echo -e "${BLUE}[INFO]${NC} 配置 ${ZSHRC} 使用 powerlevel10k 与插件..."
  touch "${ZSHRC}"

  # 设置 ZSH_THEME 为 powerlevel10k
  if grep -q '^ZSH_THEME=' "${ZSHRC}"; then
    sed -i 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' "${ZSHRC}"
  else
    echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> "${ZSHRC}"
  fi

  # 设置插件列表（保留用户已有的，若无则给一套默认）
  if grep -q '^plugins=' "${ZSHRC}"; then
    # 简单方式：如果没有我们关心的插件就追加
    for p in "git" "autojump" "zsh-autosuggestions" "zsh-syntax-highlighting" "zsh-autocomplete"; do
      if ! grep -q "$p" "${ZSHRC}"; then
        sed -i "s/^plugins=(/plugins=($p /" "${ZSHRC}" || true
      fi
    done
  else
    cat >> "${ZSHRC}" <<'EOF'
plugins=(
  git
  autojump
  zsh-autosuggestions
  zsh-syntax-highlighting
  zsh-autocomplete
)
EOF
  fi

  # 确保 zsh-syntax-highlighting 在文件末尾 source
  if ! grep -q 'zsh-syntax-highlighting.zsh' "${ZSHRC}"; then
    cat >> "${ZSHRC}" <<'EOF'

# zsh-syntax-highlighting 必须在所有插件后面加载
if [ -f "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" ]; then
  source "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh"
fi
EOF
  fi

  # 修正 .zshrc 所有者和用户组（如果组名不存在则回退到用户主组）
  local TARGET_GROUP
  TARGET_GROUP="$(id -gn "${TARGET_USER}" 2>/dev/null || echo "${TARGET_USER}")"
  chown "${TARGET_USER}:${TARGET_GROUP}" "${ZSHRC}" || true

  echo -e "${BLUE}[INFO]${NC} 将 ${TARGET_USER} 的默认 shell 更改为 zsh..."
  chsh -s "$(command -v zsh)" "${TARGET_USER}" || true

  echo -e "${GREEN}[OK]${NC} zsh / oh-my-zsh / powerlevel10k / 插件安装与配置完成"
}

# 如需启用 zsh 环境，请在 root 下执行：
#   bash nvidia.sh 之后，手动调用：
#   sudo bash -c 'source nvidia.sh; install_zsh_env'
install_zsh_env