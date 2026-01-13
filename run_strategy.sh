#!/bin/bash
# Aller dans le dossier du projet
cd /home/haithem/Documents/perso/bourse/strategy/DPMStrategy

# Définir l'heure de Paris pour le script (important pour le log)
export TZ="Europe/Paris"

# Activer l'environnement virtuel
source venv/bin/activate

# Lancer le moteur
# On utilise -u pour avoir des logs en temps réel (unbuffered)
python3 -u -m application_logic.live_engine >> logs/trading_cron.log 2>&1

