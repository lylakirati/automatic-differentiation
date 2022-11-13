#!/usr/bin/env bash

# list of test cases you want to run
tests=(
       tests/test_codes/test_dualNumber.py
       tests/test_codes/test_elementary.py
       #tests/test_codes/test_forwardAD.py
)


# we must add the module source path because we use `import cs107_package` in our test suite and we
# want to test from the source directly (not a package that we have (possibly) installed earlier)

export PYTHONPATH="$(pwd -P)/../src/":${PYTHONPATH}


# decide what driver to use (depending on arguments given)
# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python3 ${@} -m unittest"
fi

# echo ${driver} ${tests[@]}

# run the tests
${driver} ${tests[@]}

