#!/usr/bin/env bash
set -e
echo "== /healthz =="
curl -s -i http://localhost:8000/healthz
echo -e "\n\n== /predict =="
curl -s -i -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"PropertyGFATotal":250000,"NumberofFloors":12,"YearBuilt":1998,"PrimaryPropertyType":"Large Office","HasParking":true}'
echo -e "\n\n== /version =="
curl -s -i http://localhost:8000/version
echo
