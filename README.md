# Visualization for AIND ephys pipeline
## aind-ephys-visualization


### Description

This capsule is designed to visualize ephys and spike sorted data for the AIND pipeline.

Visualizations is done using the [`sortingview`](https://spikeinterface.readthedocs.io/en/latest/modules/widgets.html#id6) backend of the `spikeinterface.widgets` module and uses the [Figurl](https://github.com/flatironinstitute/figurl) technology to produce cloud-based shareable links.

Two types of visualizations are produced:

- traces: raw, preprocessed, and drift visualizations (see [example](https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://6e48d90de686e4b1422a080b6398ae6f04b8bc30&label=ecephys_664438_2023-04-12_14-59-51%20-%20experiment1_Record%20Node%20101%23Neuropix-PXI-100.probeB-AP_recording1&zone=aind))
- sorting summary: spike sorting results for visualization and curation (see [example](https://figurl.org/f?v=gs://figurl/spikesortingview-10&d=sha1://6b6e5da67a3753601b94ac23cbf2d2d31141b9e3&s={%22sortingCuration%22:%22gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main/ecephys_664438_2023-04-12_14-59-51/experiment1_Record%20Node%20101%23Neuropix-PXI-100.probeB-AP_recording1/kilosort2_5/curation.json%22}&label=ecephys_664438_2023-04-12_14-59-51%20-%20experiment1_Record%20Node%20101%23Neuropix-PXI-100.probeB-AP_recording1%20-%20kilosort2_5%20-%20Sorting%20Summary&zone=aind))


### Inputs

The `data/` folder must include:

- the original session data (e.g., "ecephys_664438_2023-04-12_14-59-51")
- the output of the [aind-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing) capsule
- the output of the spike sorting capsule (either with [aind-ephys-spikesort-pykilosort](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-pykilosort) or [aind-ephys-spikesort-kilosort25](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort25))
- the output of the [aind-ephys-postprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing) capsule
- the output of the [aind-ephys-curation](https://github.com/AllenNeuralDynamics/aind-ephys-curation) capsule (optional)
- the output of the [aind-ephys-unit-classifier](https://github.com/AllenNeuralDynamics/aind-ephys-unit-classifier) capsule (optional)

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

