#!/bin/bash

# If no arguments are provided, run pytest
if [ $# -eq 0 ]; then
  echo "No args provided, running pytest."
  pytest
else
  # Execute the mediner command with all arguments
  echo "Args provided, running mediner with args."
  mediner "$@"
fi
