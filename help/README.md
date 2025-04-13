# Motion Planning Testing Scripts

This directory contains scripts for testing motion planning algorithms with different parameters and configurations.

## Scripts

- `test.py`: Runs a single test with specified parameters
- `batch_test.py`: Runs multiple tests with different noise levels by calling `test.py` for each configuration

## Usage

### Running a Single Test

To run a single test with default parameters:

```bash
python help/test.py
```

With custom parameters:

```bash
python help/test.py --sqp_variant vanilla --num_tests 5 --num_noise_starts 2 --noise_level 0.1 --batch_name my_test
```

### Running Batch Tests

To run batch tests with default parameters (multiple noise levels):

```bash
python help/batch_test.py
```

With custom parameters:

```bash
python help/batch_test.py --sqp_variant line_search --noise_levels 0.0,0.05,0.1 --num_tests 5 --num_noise_starts 2
```

## Command-Line Arguments

Both scripts accept the following command-line arguments:

### SQP Variant
- `--sqp_variant`: SQP variant to use (choices: 'vanilla', 'line_search', 'trust_region', 'vanilla_BFGS', 'line_search_maratos')

### Test Parameters
- `--batch_name`: Name of the test batch (used for file naming)
- `--num_tests`: Number of test cases to run
- `--num_noise_starts`: Number of noisy initializations per test case
- `--noise_level`: Amount of noise to add to initial trajectories
- `--control_points`: Number of control points for noise generation
- `--smoothing_factor`: Smoothing factor for noise generation
- `--plot_indices`: Comma-separated list of test indices to generate plots for
- `--plot_initial`: Whether to plot initial trajectories

### Optimization Parameters
- `--base_set`: Base set to use (choices: 'dynamic', 'obstacle')
- `--convergence_tol`: Convergence tolerance
- `--qp_tol`: QP tolerance
- `--eq_tol`: Equality constraint tolerance
- `--ineq_tol`: Inequality constraint tolerance
- `--time_scaling`: Time scaling factor
- `--max_iter`: Maximum number of iterations
- `--initial_sigma`: Initial sigma value
- `--initial_trust_radius`: Initial trust radius
- `--initial_hessian_scale`: Initial Hessian scale (-1 for exact Hessian)

### Additional Arguments for Batch Testing
- `--noise_levels`: Comma-separated list of noise levels to test

## Output

Test results are saved in the `test_results` directory with the following structure:

- For single tests: `test_results/<batch_name>/`
- For batch tests: `test_results/batch_test_<sqp_variant>_<timestamp>/`

Each test directory contains:
- CSV files with test results
- Summary text files
- Plots (if requested)
- Configuration information

## Examples

### Example 1: Run a quick test with vanilla SQP

```bash
python help/test.py --sqp_variant vanilla --num_tests 3 --num_noise_starts 1 --noise_level 0.0
```

### Example 2: Run batch tests with line search and different noise levels

```bash
python help/batch_test.py --sqp_variant line_search --noise_levels 0.0,0.05,0.1 --num_tests 5 --num_noise_starts 2
```

### Example 3: Run a test with plots for specific test cases

```bash
python help/test.py --sqp_variant trust_region --num_tests 10 --plot_indices 0,1,2 --plot_initial
``` 