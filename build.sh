#!/bin/zsh
xhost local:
docker-compose build
docker-compose up -d
