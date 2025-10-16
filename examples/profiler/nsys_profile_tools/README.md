# gputrc2graph.py

This script processes NVIDIA Nsight Systems (`nsys`) GPU trace files
(`.nsys-rep`) with -t cuda tracing enabled, and generates kernel-level
summaries and visualizations of GPU and non-GPU time. It is useful for
profiling and analyzing nsys profile output.

## Usage

### Command-line Arguments

- `--in_file`
  **(required)**
  List of input files and their metadata. Each entry should be in the format:
  `<nsys-rep>,<engine>,<model>,<elapsed_nonprofiled_sec>`
  - `nsys-rep`: Path to the `.nsys-rep` file.
  - `engine`: Engine name (e.g., `sglang`).
  - `model`: Model name (e.g., `llama`, `gpt-oss`, `ds`).
  - `elapsed_nonprofiled_sec`: Wall-clock runtime (in seconds) without
    profiling. Specify `0` to use the elapsed time from the nsys-rep file
    (this may inflate non-GPU time if actual runtime without profiling is
    less). Multiple entries can be provided, separated by spaces.

- `--out_dir`
  Output directory for the generated CSV and HTML files.
  If not specified, results are saved in the current directory.

- `--title`
  Title for the HTML chart/visualization.

- `--nsys_cmd`
  Path to the `nsys` command.
  Default: `nsys` (assumes it is in your PATH).
  Use this if `nsys` is not in your system PATH.

## Notes

- Make sure you have pandas installed. Any version is fine.
- Make sure [nsys](https://developer.nvidia.com/nsight-systems/get-started) is
installed, and specify the path to the `nsys` command with `--nsys_cmd` if it
 is not in your PATH. The nsys version must be >= the nsys profile version that
 was used to collect the traces when profiling the server, so that nsys can
 process the nsys-rep that was generated.

- For more details on available engines and models, see the help string in
  the script or run:

```bash
python3 gputrc2graph.py --help
```

## Example 1: analyze a single profile

To analyze the GPU cycles of for example, a llama-3.1-8B model with sglang:

1. Run the following command to collect nsys profile, for sglang server config.

   ```bash
   nsys profile -t cuda -o nsys_res -f true --trace-fork-before-exec=true \
   --cuda-graph-trace=node --delay <DELAY> --duration <DURATION> \
   python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B ...
   ```

   where:

   - DELAY: how many seconds to delay nsys from collecting profiles, needed so
     that profiles aren't captured till sglang server has come up and load
     generation starts.
   - DURATION: how many seconds for nsys profile to run before generating the
     profile. This should be > the duration of the run.
2. After the server starts, run the client load generation command. Once the
test completes, after DURATION amount of time, nsys profile will generate an
nsys_res.nsys-rep file and shut down the server.

3. Run step #1 again, this time starting up the server without collecting the
profile.

4. Run step #2 again, and record the total time to complete the test in
seconds. This value will be used by the script to calculate the
   CPU(non-GPU) seconds for the analysis.

5. Say the run elapsed time from step #4 is 132 seconds. Run script to
   analyze:

   ```bash
   python3 gputrc2graph.py \
   --in_file run1.nsys-rep,sglang,llama,132
   ```

The command will produce 2 files for analysis:

- result.html: this categorizes kernel names into different categories in a
  stacked bar chart.
- result.csv: shows how the kernel names are mapped to the different
  categories.

### HTML visualization with result.html

The html file shows the number of elapsed seconds due to different GPU
Substages or categories, which consist of attention kernels as the biggest
category, at 63 seconds, followed by "gemm" kernels. This lets the user
prioritize the kernels to focus on for performance optimizations.

There's also an appended data table underneath the bar chart for copying out to
 other post-processing tools.

### Kernel to category mapping with result.csv

Suppose the user would like to focus on improving triton kernels. It's not the
biggest consumer of cycles at .01 sec but perhaps it hasn't been optimized.
The next step is to use the result.csv to dive into what the kernels are which
compose the triton kernel GPU cycles.

## Example 2: analyze multiple profiles

Suppose the user has multiple nsys trace files, captured for different models,
say llama and gpt-oss in this case, and wish to compare their GPU/non-GPU
time, something like the following command can be used.

```bash
python3 gputrc2graph.py \
--in_file run1.nsys-rep,sglang,llama,100 run2.nsys-rep,sglang,gpt-oss,102 \
--out_dir results
```

The analysis process is similar to example 1 but now there will be multiple
stack bar charts that can be compared.  The categories for the different
kernels will remain the same, so that it's easy to compare the GPU cycles for
the same categories.

Once a category is shown to have more cycles for one configuration than
another, the next step would be to use the csv file to see what kernels are
mapped into that category, and which kernels are taking the largest amount of
time which would cause a difference for the overall category.

## Example 3: add new classification for a new model

To create a new engine DEF with model ABC, just add another json file in the same directory as
gputrc2graph.py with the same format as the other json files. The script will automatically pick up all the json files in the same directory as engine/model specifications.

Then, for this new model, suppose there are 4 kernels to be classified into
"gemm" and "attn", where the gemm kernels have names with "*H*" or "*I*" in
them, and attn kernels have names with "*J*" or "*K*" in them, just add another
 .json file in the same directory as gputrc2graph.py with the same format as
 the other json files, like the following:

```json
{
  "DEF": {
      "ABC": {
          "H|I": "gemm",
          "J|K": "attn",
          "CUDA mem": "non-gpu-H_D_memops",
          ".*": "misc"
      }
  }
}
```

Each entry in the dictionary consists of:

- key: a regex used to classify the kernels
- value: the category to classify the kernels into.

The last 2 entries are common for all engine/models, consisting of CUDA memory
operations and a 'misc' for anything that's leftover and can't be classified.

When invoking gputrc2graph.py, specify a trace file with this new model/engine
like the following:

```bash
--in_file new.nsys-rep,DEF,ABC,<runtime>
```

If the engine_DEF.json file already exists, just add the model as a new node in
 the existing engine file, after the other models.
