coverage run -m pytest
cov=$(coverage report -m | tail -n 1 | awk '{print $NF}' | sed 's/.$//')
min=90
if [ "$cov" -lt "$min" ]; then
    exit 1;
fi
