"""Docker container entrypoint for task evaluation.

Contains the EvaluationRunner class and the entrypoint that invokes it.

Expects the following mount layout:
  /eval/task/     -- task directory (read-only)
  /eval/output/   -- agent output directory (read-only)
  /eval/results/  -- results directory (read-write)
  /eval/code/     -- evaluation code (evaluator/ and metrics/)
"""

import sys
import os
import json
import yaml
import importlib
import logging
import traceback
import time
import numpy as np

# Add the code directory to sys.path so metrics.* and evaluator.* are importable
sys.path.insert(0, "/eval/code")


class EvaluationRunner:
    def __init__(self, task_dir, output_dir, results_dir):
        self.task_dir = task_dir
        self.output_dir = output_dir
        self.results_dir = results_dir
        self._cache = {}

        # Load task YAML
        task_yaml_path = os.path.join(task_dir, "task.yaml")
        with open(task_yaml_path, 'r') as f:
            self.task = yaml.safe_load(f)

        self.task_id = self.task['id']

        # Set up logging
        os.makedirs(results_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(f"eval.{self.task_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler
        log_path = os.path.join(self.results_dir, "eval.log")
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(fh)

        # Stderr handler
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
        self.logger.addHandler(sh)

    def _resolve_input(self, inp):
        """Resolve a single input spec to its value."""
        source = inp['source']

        if source == 'const':
            return inp['value']
        elif source == 'output':
            output_config = self.task['output_config']
            return os.path.join(self.output_dir, output_config['file_path'])
        elif source == 'file':
            return os.path.join(self.task_dir, inp['value'])
        elif source == 'files':
            return [os.path.join(self.task_dir, v) for v in inp['value']]
        else:
            raise ValueError(f"Unsupported input source type: {source}")

    def _build_cache_key(self, step, kwargs):
        """Build a cache key from module.function and kwargs."""
        module_name = step['module']
        func_name = step['function']

        try:
            kwargs_str = json.dumps(kwargs, sort_keys=True)
        except TypeError:
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

        return f"{module_name}.{func_name}:{kwargs_str}"

    def _execute_step(self, step):
        """Execute a single evaluation step. Returns (result_value, kwargs_used)."""
        module_name = step['module']
        func_name = step['function']

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        kwargs = {}
        for inp in step['inputs']:
            arg = inp['arg']
            kwargs[arg] = self._resolve_input(inp)

        cache_key = self._build_cache_key(step, kwargs)

        if cache_key in self._cache:
            self.logger.debug(f"Cache hit for {module_name}.{func_name}")
            result = self._cache[cache_key]
        else:
            self.logger.info(f"Calling {module_name}.{func_name}({kwargs})")
            result = func(**kwargs)
            self._cache[cache_key] = result
            self.logger.info(f"Result: {result}")

        return result

    def _check_condition(self, val, condition):
        """Check a value against a condition dict. Returns bool."""
        if val is None:
            return False

        if 'eq' in condition:
            return val == condition['eq']
        if 'gt' in condition:
            return val > condition['gt']
        if 'lt' in condition:
            return val < condition['lt']
        if 'gte' in condition:
            return val >= condition['gte']
        if 'lte' in condition:
            return val <= condition['lte']

        raise ValueError(f"No recognized condition operator in: {condition}")

    def _make_json_safe(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_safe(obj.tolist())
        return obj

    def _write_error_file(self, step_idx, step_name, tb_str):
        """Write a traceback to results_dir/errors/step_<idx>_<name>.err"""
        errors_dir = os.path.join(self.results_dir, "errors")
        os.makedirs(errors_dir, exist_ok=True)
        err_path = os.path.join(errors_dir, f"step_{step_idx}_{step_name}.err")
        with open(err_path, 'w') as f:
            f.write(tb_str)

    def run(self):
        """Run all evaluation steps. Returns a result dict."""
        self.logger.info(f"Evaluating task: {self.task_id}")
        self.logger.info(f"Task dir: {self.task_dir}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Results dir: {self.results_dir}")

        steps_results = []

        for idx, step in enumerate(self.task['evaluation']):
            step_name = step['name']
            self.logger.info(f"--- Step {idx}: {step_name} ---")
            t0 = time.time()

            try:
                raw_result = self._execute_step(step)

                # Extract result_key if specified
                check_val = raw_result
                result_key = step.get('result_key')
                if result_key:
                    if isinstance(raw_result, dict):
                        check_val = raw_result[result_key]
                    else:
                        raise ValueError(
                            f"Step '{step_name}' specified result_key '{result_key}' "
                            f"but function returned {type(raw_result).__name__} instead of dict"
                        )

                passed = self._check_condition(check_val, step['condition'])
                duration = time.time() - t0

                status = "passed" if passed else "failed"
                self.logger.info(f"Step '{step_name}': {status} (result={check_val}, condition={step['condition']}, duration={duration:.2f}s)")

                steps_results.append({
                    "name": step_name,
                    "status": status,
                    "result": self._make_json_safe(check_val),
                    "condition": step['condition'],
                    "duration_s": round(duration, 2),
                })

            except Exception as e:
                duration = time.time() - t0
                tb_str = traceback.format_exc()

                self.logger.error(f"Step '{step_name}' errored: {e}")
                self.logger.error(tb_str)
                self._write_error_file(idx, step_name, tb_str)

                steps_results.append({
                    "name": step_name,
                    "status": "errored",
                    "error": str(e),
                    "error_traceback": tb_str,
                    "duration_s": round(duration, 2),
                })

        # Determine overall status: errored > failed > passed
        statuses = [s['status'] for s in steps_results]
        if 'errored' in statuses:
            overall_status = "errored"
        elif 'failed' in statuses:
            overall_status = "failed"
        else:
            overall_status = "passed"

        result = {
            "task_id": self.task_id,
            "status": overall_status,
            "steps": steps_results,
        }

        self.logger.info(f"Task {self.task_id} overall status: {overall_status}")
        return result


TASK_DIR = "/eval/task"
OUTPUT_DIR = "/eval/output"
RESULTS_DIR = "/eval/results"


def main():
    runner = EvaluationRunner(
        task_dir=TASK_DIR,
        output_dir=OUTPUT_DIR,
        results_dir=RESULTS_DIR,
    )
    result = runner.run()

    result_path = os.path.join(RESULTS_DIR, "result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Evaluation complete: {result['status']}", file=sys.stderr)


if __name__ == "__main__":
    main()
