# Set Up Steps
For HUMAN-EVAL
1. RUN:
conda create -n codex python=3.11 
conda activate codex
2. cd openai-simple-evals. 
Then RUN: 
git clone https://github.com/openai/human-eval
pip install -e human-eval
3. rename 'human-eval' to 'humaneval'
4. pip install -r requirements.txt
5. python -m demo

# Known Error
Pickle Issue: Can't pickle local object 'check_correctness.<locals>.unsafe_execute
[Issue Solution](https://github.com/openai/human-eval/pull/30)
[Issue 27](https://github.com/openai/human-eval/issues/27)
Make the above adjustment to human_eval/execution.py

