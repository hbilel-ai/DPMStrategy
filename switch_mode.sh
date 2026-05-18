#!/bin/bash
MODE=$1
BOT_DIR="ibkr-bot"

# SAFETY CHECK: Ensure files exist before touching them
if [[ ! -f "config.yaml" ]] || [[ ! -f "$BOT_DIR/docker-compose.yml" ]]; then
    echo "ERROR: Configuration files not found!"
    exit 1
fi

if [ "$MODE" == "live" ]; then
    # 1. Update config.yaml: Look for 'port:' under 'ibkr:'
    # Uses a line-range or specific match to avoid changing smtp_port
    sed -i '/ibkr:/,/port:/ s/port: .*/port: 7497/' "config.yaml"

    # 2. Update docker-compose.yml: Use the Key names as anchors
    sed -i 's/TRADING_MODE=.*/TRADING_MODE=live/' "$BOT_DIR/docker-compose.yml"
    sed -i 's/API_PORT=.*/API_PORT=4001/' "$BOT_DIR/docker-compose.yml"
    sed -i 's/SOCAT_PORT=.*/SOCAT_PORT=4003/' "$BOT_DIR/docker-compose.yml"

    # Update the port mapping line (looks for the line containing ':400')
    sed -i '/:400/ s/"[0-9]*:[0-9]*"/"7497:4003"/' "$BOT_DIR/docker-compose.yml"

    echo "Switched to LIVE mode (Bridge: 7497 -> 4003 -> 4001)"

elif [ "$MODE" == "paper" ]; then
    sed -i '/ibkr:/,/port:/ s/port: .*/port: 7496/' "config.yaml"

    sed -i 's/TRADING_MODE=.*/TRADING_MODE=paper/' "$BOT_DIR/docker-compose.yml"
    sed -i 's/API_PORT=.*/API_PORT=4002/' "$BOT_DIR/docker-compose.yml"
    sed -i 's/SOCAT_PORT=.*/SOCAT_PORT=4004/' "$BOT_DIR/docker-compose.yml"

    sed -i '/:400/ s/"[0-9]*:[0-9]*"/"7496:4004"/' "$BOT_DIR/docker-compose.yml"

    echo "Switched to PAPER mode (Bridge: 7496 -> 4004 -> 4002)"
else
    echo "Usage: ./switch_mode.sh [live|paper]"
    exit 1
fi

# Apply the changes
docker compose -f "$BOT_DIR/docker-compose.yml" down
docker compose -f "$BOT_DIR/docker-compose.yml" up -d --force-recreate
