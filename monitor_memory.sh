#!/bin/bash
# Script de monitoring mémoire Docker en temps réel

echo "=== Monitoring Mémoire Docker ==="
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

# Fichier de log
LOG_FILE="memory_monitor_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,container_name,memory_usage,memory_percent,cpu_percent" > $LOG_FILE

echo "Recherche des conteneurs sentiment..."

# Boucle de monitoring
while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Afficher tous les conteneurs Docker actifs
    STATS=$(docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}" 2>/dev/null)

    if [ -z "$STATS" ]; then
        echo "[$TIMESTAMP] Aucun conteneur actif"
    else
        echo "[$TIMESTAMP]"
        echo "$STATS" | grep -i "sentiment\|NAME"

        # Sauvegarder dans CSV (seulement les conteneurs sentiment)
        echo "$STATS" | grep -i "sentiment" | while read name mem_usage mem_perc cpu_perc; do
            echo "$TIMESTAMP,$name,$mem_usage,$mem_perc,$cpu_perc" >> $LOG_FILE
        done
    fi

    echo "---"
    sleep 2
done