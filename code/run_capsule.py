import warnings
warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import shutil
import json
import sys
import time
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.qualitymetrics as sqm
import spikeinterface.widgets as sw
from spikeinterface.core.core_tools import check_json

from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

# VIZ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sortingview.views as vv

# AIND
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess


URL = "https://github.com/AllenNeuralDynamics/aind-capsule-ephys-visualization"
VERSION = "0.1.0"

GH_CURATION_REPO = "gh://AllenNeuralDynamics/ephys-sorting-manual-curation/main"


visualization_params = dict(
    timeseries=dict(n_snippets_per_segment=2, snippet_duration_s=0.5, skip=False),
    drift=dict(detection=dict(method='locally_exclusive', peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1), 
               localization=dict(ms_before=0.1, ms_after=0.3, local_radius_um=100.),
               n_skip=30, alpha=0.15, vmin=-200, vmax=0, cmap="Greys_r",
               figsize=(10, 10))
)

job_kwargs = {
    'n_jobs': -1,
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
results_folder = Path("../results/")


if __name__ == "__main__":
    data_processes_folder = results_folder / "data_processes_visualization"
    data_processes_folder.mkdir(exist_ok=True)
    
    si.set_global_job_kwargs(**job_kwargs)

    ###### VISUALIZATION #########
    print("\n\nVISUALIZATION")
    t_visualization_start = time.perf_counter()
    datetime_start_visualization = datetime.now()
    visualization_output = {}

    # check if test
    if (data_folder / "postprocessing_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_output_test" / "postprocessed"
        preprocessed_folder = data_folder / "preprocessing_output_test" / "preprocessed"
        visualization_folder = data_folder / "preprocessing_output_test" / "visualization_preprocessed"
        sorting_precurated_folder = data_folder / "curation_output_test" / "sorting_precurated"
        data_processes_spikesorting_folder =  data_folder / "spikesorting_output_test" / "data_processes_spikesorting"
    else:
        postprocessed_folder = data_folder / "postprocessed"
        preprocessed_folder = data_folder / "preprocessed"
        visualization_folder = data_folder / "visualization_preprocessed"
        sorting_precurated_folder = data_folder / "sorting_precurated"
        data_processes_spikesorting_folder =  data_folder / "data_processes_spikesorting"

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    print(f"Sessions: {ecephys_sessions}")
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    # loop through block-streams
    for recording_folder in preprocessed_folder.iterdir():
        recording_name = recording_folder.name
        waveforms_folder = postprocessed_folder / recording_name
        print(f"Visualizing recording: {recording_name}")

        # retrieve sorter name
        with open(data_processes_spikesorting_folder / f"spikesorting_{recording_name}.json", "r") as f:
            data_process_spikesorting = json.load(f)
            sorter_name = data_process_spikesorting["parameters"]["sorter_name"]

        with open(visualization_folder / f"{recording_name}.json", "r") as f:
            preprocessing_vizualization_data = json.load(f)
        
        if recording_name not in visualization_output:
            visualization_output[recording_name] = {}
        recording_processed = si.load_extractor(recording_folder)

        # drift
        cmap = plt.get_cmap(visualization_params["drift"]["cmap"])
        norm = Normalize(vmin=visualization_params["drift"]["vmin"], vmax=visualization_params["drift"]["vmax"], clip=True)
        n_skip = visualization_params["drift"]["n_skip"]
        alpha = visualization_params["drift"]["alpha"]

        # use spike locations
        if waveforms_folder.is_dir():
            print(f"\tVisualizing drift maps using spike sorted data")
            we = si.load_waveforms(waveforms_folder, with_recording=False)
            recording = recording_processed
            peaks = we.sorting.to_spike_vector()
            peak_locations = we.load_extension("spike_locations").get_data()
            peak_amps = np.concatenate(we.load_extension("spike_amplitudes").get_data())
        # otherwise etect peaks
        else:

            print(f"\tVisualizing drift maps using detected peaks (no spike sorting available)")
            # locally_exclusive + pipeline steps LocalizeCenterOfMass + PeakToPeakFeature
            drift_data = preprocessing_vizualization_data[recording_name]["drift"]
            recording = si.load_extractor(drift_data["recording"], base_folder=".")
            extract_dense_waveforms = ExtractDenseWaveforms(recording, ms_before=visualization_params["drift"]["localization"]["ms_before"],
                                                            ms_after=visualization_params["drift"]["localization"]["ms_after"], return_output=False)
            localize_peaks = LocalizeCenterOfMass(recording, local_radius_um=visualization_params["drift"]["localization"]["local_radius_um"], 
                                                  parents=[extract_dense_waveforms])
            pipeline_nodes = [
                extract_dense_waveforms,
                localize_peaks
            ]
            peaks, peak_locations = detect_peaks(recording,
                                                 pipeline_nodes=pipeline_nodes,
                                                 **visualization_params["drift"]["detection"])
            print(f"\tDetected {len(peaks)} peaks")
            peak_amps = peaks["amplitude"]

        y_locs = recording.get_channel_locations()[:, 1]
        ylim = [np.min(y_locs), np.max(y_locs)]

        fig_drift, axs_drift = plt.subplots(ncols=recording.get_num_segments(), figsize=visualization_params["drift"]["figsize"])
        for segment_index in range(recording.get_num_segments()):
            segment_mask = peaks["segment_ind"] == segment_index
            x = peaks[segment_mask]['sample_ind'] / recording.sampling_frequency
            y = peak_locations[segment_mask]['y']
            # subsample
            x_sub = x[::n_skip]
            y_sub = y[::n_skip]
            a_sub = peak_amps[::n_skip]
            colors = cmap(norm(a_sub))

            if recording.get_num_segments() == 1:
                ax_drift = axs_drift
            else:
                ax_drift = axs_drift[segment_index]
            ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
            ax_drift.set_xlabel("time (s)", fontsize=12)
            ax_drift.set_ylabel("depth ($\\mu$m)", fontsize=12)
            ax_drift.set_xlim(0, recording.get_num_samples(segment_index=segment_index) / recording.sampling_frequency)
            ax_drift.set_ylim(ylim)
            ax_drift.spines["top"].set_visible(False)
            ax_drift.spines["right"].set_visible(False)
        fig_drift_folder = results_folder / "drift_maps"
        fig_drift_folder.mkdir(exist_ok=True)
        fig_drift.savefig(fig_drift_folder / f"{recording_name}_drift.png", dpi=300)

        # make a sorting view View
        v_drift = vv.TabLayoutItem(
            label=f"Drift map",
            view=vv.Image(image_path=str(fig_drift_folder / f"{recording_name}_drift.png"))
        )

        # timeseries
        if not visualization_params["timeseries"]["skip"]:
            timeseries_tab_items = []
            print(f"\tVisualizing timeseries")

            timeseries_data = preprocessing_vizualization_data[recording_name]["timeseries"]
            recording_full_dict = timeseries_data["full"]
            recording_proc_dict = timeseries_data["proc"]

            # get random chunks to estimate clims
            clims_full = {}
            recording_full_loaded = {}
            for layer, rec_dict in recording_full_dict.items():
                try:
                    rec = si.load_extractor(rec_dict, base_folder=".")
                except:
                    print(f"\t\tCould not load layer {layer}. Skipping")
                    continue
                chunk = si.get_random_data_chunks(rec)
                max_value = np.quantile(chunk, 0.99) * 1.2
                clims_full[layer] = (-max_value, max_value)
                recording_full_loaded[layer] = rec
            clims_proc = {}
            if recording_proc_dict is not None:
                recording_proc_loaded = {}
                for layer, rec_dict in recording_proc_dict.items():
                    try:
                        rec = si.load_extractor(rec_dict, base_folder=".")
                    except:
                        print(f"\t\tCould not load layer {layer}. Skipping")
                        continue
                    chunk = si.get_random_data_chunks(rec)
                    max_value = np.quantile(chunk, 0.99) * 1.2
                    clims_proc[layer] = (-max_value, max_value)
                    recording_proc_loaded[layer] = rec
            else:
                print(f"\tPreprocessed timeseries not avaliable")

            fs = recording.get_sampling_frequency()
            n_snippets_per_seg = visualization_params["timeseries"]["n_snippets_per_segment"]
            try:
                for segment_index in range(recording.get_num_segments()):
                    segment_duration = recording.get_num_samples(segment_index) / fs
                    # evenly distribute t_starts across segments
                    t_starts = np.linspace(0, segment_duration, n_snippets_per_seg + 2)[1:-1]
                    for t_start in t_starts:
                        time_range = np.round(np.array([t_start, t_start + visualization_params["timeseries"]["snippet_duration_s"]]), 1)
                        w_full = sw.plot_timeseries(recording_full_loaded, order_channel_by_depth=True, time_range=time_range, 
                                                    segment_index=segment_index, clim=clims_full, backend="sortingview", generate_url=False)
                        if recording_proc_dict is not None:
                            w_proc = sw.plot_timeseries(recording_proc_loaded, order_channel_by_depth=True, time_range=time_range, 
                                                        segment_index=segment_index, clim=clims_proc, backend="sortingview", generate_url=False)
                            view = vv.Splitter(
                                        direction='horizontal',
                                        item1=vv.LayoutItem(w_full.view),
                                        item2=vv.LayoutItem(w_proc.view)
                                    )
                        else:
                            view = w_full.view
                        v_item = vv.TabLayoutItem(
                            label=f"Timeseries - Segment {segment_index} - Time: {time_range}",
                            view=view
                        )
                        timeseries_tab_items.append(v_item)
                # add drift map
                timeseries_tab_items.append(v_drift)

                v_timeseries = vv.TabLayout(
                    items=timeseries_tab_items
                )
                try:
                    url = v_timeseries.url(label=f"{session_name} - {recording_name}")
                    print(f"\n{url}\n")
                    visualization_output[recording_name]["timeseries"] = url
                except Exception as e:
                    print("KCL error", e)
            except Exception as e:
                print(f"Something wrong when visualizing timeseries: {e}")

        # sorting summary        
        print(f"\tVisualizing sorting summary")        
        we = si.load_waveforms(waveforms_folder, with_recording=False)
        we._recording = recording_processed
        sorting_precurated = si.load_extractor(sorting_precurated_folder / recording_name)
        # set waveform_extractor sorting object to have pass_qc property
        we.sorting = sorting_precurated

        
        if len(we.sorting.unit_ids) > 0:
            # tab layout with Summary and Quality Metrics
            v_qm = sw.plot_quality_metrics(we, skip_metrics=['isi_violations_count', 'rp_violations'], 
                                           include_metrics_data=True, backend="sortingview", generate_url=False).view
            v_sorting = sw.plot_sorting_summary(we, unit_table_properties=["default_qc"], curation=True, 
                                                backend="sortingview", generate_url=False).view

            v_summary = vv.TabLayout(
                    items=[
                        vv.TabLayoutItem(
                            label='Sorting summary',
                            view=v_sorting
                        ),
                        vv.TabLayoutItem(
                            label='Quality Metrics',
                            view=v_qm
                        )
                    ]
                )

            try:
                # pre-generate gh for curation
                gh_path = f"{GH_CURATION_REPO}/{session_name}/{recording_name}/{sorter_name}/curation.json"
                state = dict(sortingCuration=gh_path)
                url = v_summary.url(label=f"{session_name} - {recording_name} - {sorter_name} - Sorting Summary", state=state)
                print(f"\n{url}\n")
                visualization_output[recording_name]["sorting_summary"] = url

            except Exception as e:
                print("KCL error", e)
        else:
            print("No units after curation!")

    # save params in output
    visualization_notes = json.dumps(visualization_output, indent=4)
    # replace special characters
    visualization_notes = visualization_notes.replace('\\"', "%22")
    visualization_notes = visualization_notes.replace('#', "%23")

    # save vizualization output
    visualization_output_file = results_folder / "visualization_output.json"
    # remove escape characters
    visualization_output_file.write_text(visualization_notes)

    # save vizualization output
    t_visualization_end = time.perf_counter()
    elapsed_time_visualization = np.round(t_visualization_end - t_visualization_start, 2)

    visualization_process = DataProcess(
            name="Ephys visualization",
            version=VERSION, # either release or git commit
            start_date_time=datetime_start_visualization,
            end_date_time=datetime_start_visualization + timedelta(seconds=np.floor(elapsed_time_visualization)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=visualization_params,
            notes=visualization_notes
        )
    print(f"VISUALIZATION time: {elapsed_time_visualization}s")
