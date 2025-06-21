#!/bin/bash

echo "Kafka Monitoring Tools"
echo "====================="

case "$1" in
    "topics")
        echo "Listing all topics:"
        docker exec -it kafka kafka-topics --list --bootstrap-server localhost:19092
        ;;
    
    "consumer-groups")
        echo "Listing consumer groups:"
        docker exec -it kafka kafka-consumer-groups --list --bootstrap-server localhost:19092
        ;;
    
    "lag")
        echo "Checking consumer lag:"
        docker exec -it kafka kafka-consumer-groups --bootstrap-server localhost:19092 --describe --group print-consumer-group
        ;;
    
    "messages")
        TOPIC="${2:-alpaca_market_data_paper_trades}"
        echo "Last 10 messages from $TOPIC:"
        docker exec -it kafka kafka-console-consumer \
            --bootstrap-server localhost:19092 \
            --topic $TOPIC \
            --from-beginning \
            --max-messages 10 \
            --property print.timestamp=true \
            --property print.key=true \
            --property print.value=true
        ;;
    
    "stats")
        echo "Topic statistics:"
        docker exec -it kafka bash -c '
            for topic in $(kafka-topics --list --bootstrap-server localhost:19092 | grep alpaca); do
                echo "Topic: $topic"
                kafka-run-class kafka.tools.GetOffsetShell \
                    --broker-list localhost:19092 \
                    --topic $topic \
                    --time -1 | awk -F: "{sum+=\$3} END {print \"  Total messages: \" sum}"
                echo ""
            done
        '
        ;;
    
    "monitor")
        TOPIC="${2:-alpaca_market_data_paper_trades}"
        echo "Real-time monitoring of $TOPIC (Ctrl+C to stop):"
        docker exec -it kafka kafka-console-consumer \
            --bootstrap-server localhost:19092 \
            --topic $TOPIC \
            --property print.timestamp=true \
            --property print.key=true
        ;;
    
    *)
        echo "Usage: $0 {topics|consumer-groups|lag|messages|stats|monitor} [topic]"
        echo ""
        echo "  topics         - List all Kafka topics"
        echo "  consumer-groups - List all consumer groups"
        echo "  lag            - Check consumer lag for print-consumer-group"
        echo "  messages       - Show last 10 messages from a topic"
        echo "  stats          - Show message counts for all topics"
        echo "  monitor        - Real-time monitoring of a topic"
        echo ""
        echo "Examples:"
        echo "  $0 topics"
        echo "  $0 lag"
        echo "  $0 messages alpaca_market_data_paper_quotes"
        echo "  $0 monitor alpaca_market_data_paper_trades"
        ;;
esac