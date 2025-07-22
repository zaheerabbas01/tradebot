#!/bin/bash
"""
Deployment Script for Crypto Trading Bot
Automates VPS setup and bot deployment
"""

set -e  # Exit on any error

echo "🚀 Crypto Trading Bot Deployment Script"
echo "========================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons"
   exit 1
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+ if not present
echo "🐍 Installing Python 3.9+..."
sudo apt install -y python3.9 python3.9-pip python3.9-venv python3.9-dev

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libjpeg-dev \
    libfreetype6-dev liblcms2-dev libopenjp2-7-dev \
    libtiff5-dev tk-dev tcl-dev git curl wget \
    supervisor nginx redis-server

# Install TA-Lib (required for technical indicators)
echo "📊 Installing TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr/local
make
sudo make install
sudo ldconfig
cd ~

# Create project directory
PROJECT_DIR="$HOME/crypto-trading-bot"
echo "📁 Setting up project directory: $PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create Python virtual environment
echo "🔧 Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p backtest_results
mkdir -p config

# Copy environment template
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "✏️ Please edit .env file with your API keys and configuration"
fi

# Set up log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/crypto-trading-bot > /dev/null <<EOF
$PROJECT_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $(whoami) $(whoami)
}
EOF

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/crypto-trading-bot.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading Bot
After=network.target
Wants=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/crypto_trading_bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto-trading-bot

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$PROJECT_DIR

[Install]
WantedBy=multi-user.target
EOF

# Create backup script
echo "💾 Creating backup script..."
tee backup.sh > /dev/null <<EOF
#!/bin/bash
# Backup script for crypto trading bot

BACKUP_DIR="$PROJECT_DIR/backups"
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="crypto_bot_backup_\$DATE.tar.gz"

mkdir -p "\$BACKUP_DIR"

# Create backup
tar -czf "\$BACKUP_DIR/\$BACKUP_FILE" \\
    --exclude='venv' \\
    --exclude='__pycache__' \\
    --exclude='*.pyc' \\
    --exclude='.git' \\
    "$PROJECT_DIR"

echo "Backup created: \$BACKUP_DIR/\$BACKUP_FILE"

# Keep only last 7 backups
cd "\$BACKUP_DIR"
ls -t crypto_bot_backup_*.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed successfully"
EOF

chmod +x backup.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
tee monitor.sh > /dev/null <<EOF
#!/bin/bash
# Monitoring script for crypto trading bot

echo "=== Crypto Trading Bot Status ==="
echo "Date: \$(date)"
echo ""

# Check if service is running
if systemctl is-active --quiet crypto-trading-bot; then
    echo "✅ Service Status: RUNNING"
else
    echo "❌ Service Status: STOPPED"
fi

# Check memory usage
MEMORY_USAGE=\$(ps -o pid,ppid,cmd,%mem --sort=-%mem -C python | head -2)
echo ""
echo "Memory Usage:"
echo "\$MEMORY_USAGE"

# Check disk space
echo ""
echo "Disk Usage:"
df -h "$PROJECT_DIR"

# Check recent logs
echo ""
echo "Recent Logs (last 10 lines):"
tail -n 10 "$PROJECT_DIR/logs/trading_bot_\$(date +%Y%m%d).log" 2>/dev/null || echo "No log file found"

# Check network connectivity to Binance
echo ""
echo "Network Connectivity:"
PING_RESULT=\$(ping -c 1 api.binance.com 2>/dev/null | grep 'time=' | awk -F'time=' '{print \$2}' | awk '{print \$1}')
if [ ! -z "\$PING_RESULT" ]; then
    echo "✅ Binance API: \$PING_RESULT"
else
    echo "❌ Binance API: Connection failed"
fi
EOF

chmod +x monitor.sh

# Create crontab for monitoring and backups
echo "⏰ Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * $PROJECT_DIR/backup.sh") | crontab -
(crontab -l 2>/dev/null; echo "*/5 * * * * $PROJECT_DIR/monitor.sh >> $PROJECT_DIR/logs/monitor.log 2>&1") | crontab -

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable crypto-trading-bot

# Create firewall rules (optional)
echo "🔥 Configuring firewall..."
sudo ufw --force enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp  # HTTP (if needed for monitoring)
sudo ufw allow 443/tcp # HTTPS (if needed for monitoring)

# Performance optimizations
echo "⚡ Applying performance optimizations..."

# Increase file descriptor limits
echo "$(whoami) soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "$(whoami) hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Network optimizations for low latency
sudo tee -a /etc/sysctl.conf > /dev/null <<EOF
# Network optimizations for trading bot
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.ipv4.tcp_congestion_control = bbr
EOF

sudo sysctl -p

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "Next Steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Run backtests to validate strategies:"
echo "   source venv/bin/activate"
echo "   python run_backtest.py --symbol BTC/USDT --start 2021-01-01 --end 2023-12-31 --plot --report"
echo ""
echo "3. Start the bot service:"
echo "   sudo systemctl start crypto-trading-bot"
echo ""
echo "4. Check service status:"
echo "   sudo systemctl status crypto-trading-bot"
echo ""
echo "5. View logs:"
echo "   tail -f logs/trading_bot_$(date +%Y%m%d).log"
echo ""
echo "6. Monitor the bot:"
echo "   ./monitor.sh"
echo ""
echo "🚨 IMPORTANT: Test thoroughly with small amounts before scaling up!"
echo "🔒 SECURITY: Ensure your VPS is properly secured and API keys are protected!"

deactivate