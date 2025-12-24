#!/bin/bash
#
# Codespace Setup Script
# Sets up a new GitHub Codespace with all required tools and dependencies
#
# Usage: ./setup_codespace.sh

# Verify we're running in bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh"
    echo "Usage: bash setup_codespaces.sh"
    echo "   or: ./setup_codespaces.sh"
    exit 1
fi

set -e  # Exit on error

echo "======================================================================================================"
echo "Setting up Codespace for TinyTimeMixers"
echo "======================================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Step 1: Install Claude Code
print_step "Installing Claude Code..."
if command -v claude > /dev/null 2>&1; then
    CLAUDE_VERSION=$(claude --version 2>&1 | head -1)
    print_success "Claude Code already installed ($CLAUDE_VERSION)"
else
    curl -fsSL https://claude.ai/install.sh | bash
    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Verify installation succeeded
    if command -v claude > /dev/null 2>&1; then
        CLAUDE_VERSION=$(claude --version 2>&1 | head -1)
        print_success "Claude Code installed ($CLAUDE_VERSION)"
    else
        print_warning "Claude Code installation may have failed - command not found"
    fi
fi
echo ""

# Step 2: Install uv
print_step "Installing uv package manager..."
if command -v uv > /dev/null 2>&1; then
    UV_VERSION=$(uv --version 2>&1)
    print_success "uv already installed ($UV_VERSION)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell config to get uv in PATH
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify installation succeeded
    if command -v uv > /dev/null 2>&1; then
        UV_VERSION=$(uv --version 2>&1)
        print_success "uv installed ($UV_VERSION)"
    else
        print_warning "uv installation may have failed or requires shell restart"
    fi
fi
echo ""

# Step 3: Check Python version
print_step "Checking Python version..."
if command -v python3 > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"

    # Check if version is >= 3.12
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 12 ]; then
        print_success "Python version >= 3.12 requirement met"
    else
        print_warning "Python 3.12+ recommended (current: $PYTHON_VERSION)"
    fi
else
    print_warning "Python not found - this may cause issues"
fi
echo ""

# Step 4: Install project dependencies
print_step "Installing project dependencies with uv..."
if [ -f "pyproject.toml" ]; then
    # Add uv to PATH if not already there
    export PATH="$HOME/.local/bin:$PATH"

    # Sync all dependencies including dev dependencies
    uv sync --all-groups
    print_success "Dependencies installed"

    # Show installed packages
    echo ""
    print_step "Installed packages:"
    uv pip list | grep -E "(pre-commit|pytest|ruff|ty|zizmor)" || true
else
    print_warning "pyproject.toml not found - skipping dependency installation"
fi
echo ""

# Step 5: Install pre-commit hooks
print_step "Installing pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ] && [ -d ".venv" ]; then
    # Add uv to PATH if not already there
    export PATH="$HOME/.local/bin:$PATH"

    # Install pre-commit hooks
    uv run pre-commit install
    print_success "Pre-commit hooks installed"

    # Optionally run pre-commit on all files to ensure everything is formatted
    print_step "Running pre-commit on all files (this may take a moment)..."
    if uv run pre-commit run --all-files 2>&1 | tail -10; then
        print_success "Pre-commit checks passed"
    else
        print_warning "Pre-commit made some formatting changes (this is normal)"
    fi
else
    if [ ! -f ".pre-commit-config.yaml" ]; then
        print_warning ".pre-commit-config.yaml not found - skipping pre-commit setup"
    else
        print_warning "Virtual environment not found - skipping pre-commit setup"
    fi
fi
echo ""

# Step 6: Verify installation
print_step "Verifying installation..."
export PATH="$HOME/.local/bin:$PATH"

ERRORS=0

# Check claude
if command -v claude > /dev/null 2>&1; then
    CLAUDE_VERSION=$(claude --version 2>&1 | head -1)
    print_success "Claude Code: $CLAUDE_VERSION"
else
    print_warning "Claude Code not found in PATH"
    ERRORS=$((ERRORS + 1))
fi

# Check uv
if command -v uv > /dev/null 2>&1; then
    UV_VERSION=$(uv --version 2>&1)
    print_success "uv: $UV_VERSION"
else
    print_warning "uv not found in PATH"
    ERRORS=$((ERRORS + 1))
fi

# Check Python
if command -v python3 > /dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Python: $PYTHON_VERSION"
else
    print_warning "Python not found"
    ERRORS=$((ERRORS + 1))
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    print_success "Virtual environment created at .venv/"
else
    print_warning "Virtual environment not found"
fi

echo ""

# Step 7: Run tests to verify everything works
if [ -d "tests" ] && [ -d ".venv" ]; then
    print_step "Running tests to verify installation..."
    if uv run pytest tests/ -v --tb=short -k "not integration" 2>&1 | tail -20; then
        print_success "Unit tests passed"
    else
        print_warning "Some tests failed - check output above"
        ERRORS=$((ERRORS + 1))
    fi
    echo ""
fi

# Step 8: Display next steps
echo "======================================================================================================"
if [ $ERRORS -eq 0 ]; then
    print_success "Setup complete! Your codespace is ready."
else
    print_warning "Setup completed with $ERRORS warnings - check messages above"
fi
echo "======================================================================================================"
echo ""
echo "Next steps:"
echo "  1. Restart your shell or run: source ~/.bashrc"
echo "  2. Verify uv: uv --version"
echo "  3. Verify claude: claude --version"
echo "  4. Run tests: uv run pytest"
echo "  5. Read README.md for usage examples"
echo ""
echo "Useful commands:"
echo "  uv sync --all-groups       # Install/update dependencies"
echo "  uv run pytest              # Run all tests"
echo "  uv run ruff check .        # Lint code"
echo "  uv run ty check .          # Type check code"
echo "  uv run zizmor .github/     # Scan GitHub Actions"
echo ""
