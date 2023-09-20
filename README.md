# Visualization for AIND ephys pipeline
## aind-capsule-ephys-visualization


### Description

This capsule is designed to visualize ephys and spike sorted data for the AIND pipeline.

Visualizations is done using the [`sortingview`]() backend of the `spikeinterface.widgets` module and uses the [Figurl]() technology to produce cloud-based shareable links.

Two types of visualizations are produced:

- traces: raw, preprocessed, and drift visualizations (see [example]())
- sorting summary: spike sorting results for visualization and curation (see [example]())


### Inputs

The `data/` folder must include:

- the original session data (e.g., "ecephys_664438_2023-04-12_14-59-51")
- the output of the [aind-capsule-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-preprocessing) capsule
- the output of the spike sorting capsule (either with [aind-capsule-ephys-spikesort-pykilosort](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-pykilosort) or [aind-capsule-ephys-spikesort-kilosort25](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-spikesort-kilosort25))
- the output of the [aind-capsule-ephys-postprocessing](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-postprocessing) capsule
- the output of the [aind-capsule-ephys-curation](https://github.com/AllenNeuralDynamics/aind-capsule-ephys-curation) capsule

### Parameters

The `code/run` script takes no arguments.

A list of visualization parameters can be found at the top of the `code/run_capsule.py` script:

```python
visualization_params = dict(
    timeseries=dict(n_snippets_per_segment=2, snippet_duration_s=0.5, skip=False),
    drift=dict(detection=dict(method='locally_exclusive', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1), 
               localization=dict(ms_before=0.1, ms_after=0.3, local_radius_um=100.),
               n_skip=30, alpha=0.15, vmin=-200, vmax=0, cmap="Greys_r",
               figsize=(10, 10)),
    motion=dict(cmap="Greys_r", scatter_decimate=15, figsize=(15, 10))
)
```

### Output

The output of this capsule is the following:

- `results/visualization_{recording_name}.json`, a JSON file including the visualization links
- `results/data_process_postprocessing_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

