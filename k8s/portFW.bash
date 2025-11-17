#!/bin/bash

echo "========================================="
echo "  Port Forwarding 4 NexusML Services"
echo "========================================="

# Namespace của bạn, nếu là default thì để default
NS="default"

# Services và ports: <service>:<local-port>:<pod-port>
declare -A PORTS
PORTS=(
    ["backend-service"]="8000:8000"
    ["frontend-service"]="3000:3000"
    ["minio-service-api"]="9000:9000"
    ["minio-service-console"]="9001:9001"
    ["mlflow-service"]="5000:5000"
)

declare -A PIDS

# Kiểm tra service tồn tại
service_exists() {
    kubectl get svc "$1" -n "$NS" >/dev/null 2>&1
}

echo "Checking services..."
for svc in "${!PORTS[@]}"; do
    if service_exists $svc; then
        echo "Found: $svc"
    else
        echo "Error: $svc not found"
    fi
done

echo ""
echo "Starting port forwarding..."
echo ""

for svc in "${!PORTS[@]}"; do
    port_map=${PORTS[$svc]}
    kubectl port-forward -n $NS service/$svc $port_map --address=0.0.0.0 >/dev/null 2>&1 &
    pid=$!
    PIDS["$svc"]=$pid
    local_port=$(echo $port_map | cut -d':' -f1)
    echo "$svc → localhost:$local_port"
done

echo ""
echo "========================================="
echo "  All services are now accessible via localhost!"
echo "========================================="
echo "Press Ctrl+C to stop all port forwarding."
echo ""

# Cleanup khi Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all port forwarding..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "Done!"
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
